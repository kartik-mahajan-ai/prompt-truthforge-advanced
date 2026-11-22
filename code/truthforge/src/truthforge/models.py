"""Text generators.

Everything downstream takes a plain ``generate_fn: str -> str``, so the loop,
the splitter and the rewriter never import torch. Three implementations:

  * ``LocalHFGenerator`` -- a decoder-only chat model on the local GPU.
  * ``EchoGenerator``    -- deterministic stub. Makes the end-to-end loop
                            runnable with no models installed, and makes the
                            loop's own logic testable without a GPU in the way.
  * ``ScriptedGenerator`` -- replays fixed outputs, for unit tests that need a
                            specific rewrite.

Trap #2 from the brief: whatever the corrector is, it must not be the judge.
The judge is Flan-T5-based MiniCheck; keep the corrector decoder-only.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from .config import GeneratorConfig


class EchoGenerator:
    """Returns a trivial, deterministic transformation of the prompt.

    For a rewrite prompt it returns the sentence unchanged, which the outcome
    classifier scores as SAME. That is the honest stub behaviour: a mock must
    not manufacture BETTER outcomes.
    """

    name = "echo"

    def __call__(self, prompt: str) -> str:
        marker = "SENTENCE TO CORRECT:"
        if marker in prompt:
            tail = prompt.split(marker, 1)[1]
            return tail.split("\n\nRules:", 1)[0].strip()
        return ""


class ScriptedGenerator:
    """Replays a queue of outputs; raises when it runs dry."""

    name = "scripted"

    def __init__(self, outputs: Sequence[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.outputs:
            raise RuntimeError("ScriptedGenerator exhausted")
        return self.outputs.pop(0)


class LocalHFGenerator:
    """A local instruction-tuned model.

    Defaults target an 8 GB card: Qwen2.5-3B-Instruct at fp16 uses ~6 GB and
    leaves room for MiniCheck-Flan-T5-Large (~1.6 GB) alongside it. A 7B model
    needs ``load_in_4bit=True`` and bitsandbytes, and will not co-reside with
    the judge on 8 GB without offloading -- run the two in separate passes
    instead, and say so in the latency column's footnote.
    """

    name = "local_hf"

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        from ._hf import load_kwargs, resolve_device, with_dtype_fallback

        c = self.cfg
        device = resolve_device(c.device)
        kwargs = load_kwargs(device, c.dtype)
        if c.load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(c.model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = with_dtype_fallback(
            AutoModelForCausalLM.from_pretrained, c.model, **kwargs
        )
        if not c.load_in_4bit:
            self._model = self._model.to(device)
        self._model.eval()
        self.device = device

    def __call__(self, prompt: str) -> str:
        self._ensure_loaded()
        import torch  # type: ignore

        c = self.cfg
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=c.max_new_tokens,
                do_sample=c.temperature > 0,
                temperature=c.temperature if c.temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tokenizer.encode(text))


def build_generator(name: str, cfg: Optional[GeneratorConfig] = None) -> Callable[[str], str]:
    if name in ("echo", "mock"):
        return EchoGenerator()
    if name in ("local", "hf", "local_hf"):
        return LocalHFGenerator(cfg)
    raise ValueError(f"unknown generator: {name!r}")
