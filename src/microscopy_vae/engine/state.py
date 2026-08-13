from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainerState:
    epoch: int = 0
    microbatch: int = 0
    optimizer_step: int = 0
    global_samples: int = 0
    best_metric: Optional[float] = None
    candidate_hits: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainerState":
        return TrainerState(
            epoch=int(d.get("epoch", 0)),
            microbatch=int(d.get("microbatch", 0)),
            optimizer_step=int(d.get("optimizer_step", 0)),
            global_samples=int(d.get("global_samples", 0)),
            best_metric=d.get("best_metric"),
            candidate_hits={int(k): v for k, v in d.get("candidate_hits", {}).items()},
        )
