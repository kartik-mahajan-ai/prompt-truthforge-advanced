"""Small compatibility helpers for the HuggingFace stack.

Isolated here so that a transformers API change is a one-file fix rather than a
grep across every model wrapper. Nothing in this module imports torch at
import time.

Known moving parts:

* ``torch_dtype`` was renamed to ``dtype`` in transformers 5.x, and the old
  name emits a deprecation warning that will eventually become an error.
* ``device_map`` handling differs between quantised and unquantised loads.
"""

from __future__ import annotations

import functools
from typing import Any, Callable


@functools.lru_cache(maxsize=1)
def _dtype_kwarg() -> str:
    """Which keyword this transformers version wants for the weight dtype."""
    try:
        import transformers  # type: ignore

        major = int(str(transformers.__version__).split(".", 1)[0])
    except Exception:  # noqa: BLE001
        return "dtype"
    return "dtype" if major >= 5 else "torch_dtype"


def resolve_device(requested: str = "cuda") -> str:
    """Honour the request when CUDA is really there, else fall back to CPU.

    Silently running on CPU is acceptable for correctness but ruinous for the
    latency column, so callers that report timings should check the returned
    value rather than assume.
    """
    import torch  # type: ignore

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def dtype_for(device: str, name: str = "float16"):
    """fp16 on GPU, fp32 on CPU.

    fp16 on CPU is both slower and less accurate on most x86 hardware, so the
    requested dtype is deliberately overridden there.
    """
    import torch  # type: ignore

    if device == "cpu":
        return torch.float32
    return getattr(torch, name)


def load_kwargs(device: str, dtype_name: str = "float16", **extra: Any) -> dict[str, Any]:
    """Build `from_pretrained` kwargs with the right dtype keyword."""
    return {_dtype_kwarg(): dtype_for(device, dtype_name), **extra}


def with_dtype_fallback(loader: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a `from_pretrained` that may want either dtype keyword.

    Belt and braces for the version boundary: if the chosen keyword is
    rejected, swap it and retry once rather than failing the run.
    """
    try:
        return loader(*args, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        swapped = dict(kwargs)
        if "dtype" in swapped:
            swapped["torch_dtype"] = swapped.pop("dtype")
        elif "torch_dtype" in swapped:
            swapped["dtype"] = swapped.pop("torch_dtype")
        else:
            raise
        return loader(*args, **swapped)


def cuda_memory_summary() -> dict[str, Any]:
    """VRAM snapshot, for the UI status panel and for OOM post-mortems."""
    try:
        import torch  # type: ignore
    except ImportError:
        return {"available": False, "reason": "torch not installed"}

    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}

    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_mb": round(total / 1024**2),
        "free_mb": round(free / 1024**2),
        "used_mb": round((total - free) / 1024**2),
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
    }


def empty_cache() -> None:
    """Release cached VRAM. Needed when swapping models on an 8 GB card."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
