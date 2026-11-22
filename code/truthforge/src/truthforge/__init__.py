"""TruthForge: deciding when correcting a RAG answer is worth it.

The research question is not "can we detect and fix hallucinations" -- that is
settled. It is *when is fixing worth it*. The router in `truthforge.router` is
the contribution; everything else feeds it or measures it.

Two pre-registered decision gates live in DECISIONS.md. Read them before
reporting any result.
"""

from .config import Config
from .cost import CostLedger
from .types import (
    Action,
    Claim,
    CorrectionAttempt,
    CostRecord,
    Label,
    Outcome,
    Passage,
    RoutingDecision,
    TurnResult,
    Verdict,
    WorseKind,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Claim",
    "Config",
    "CorrectionAttempt",
    "CostLedger",
    "CostRecord",
    "Label",
    "Outcome",
    "Passage",
    "RoutingDecision",
    "TurnResult",
    "Verdict",
    "WorseKind",
    "__version__",
]
