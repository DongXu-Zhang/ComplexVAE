from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch


@dataclass(frozen=True)
class HQPageRecord:
    sample_id: str
    split: Literal["train", "val", "test"]
    source: str
    category: str
    condition: str
    morphology: str
    group_id: str
    hq_path: Path
    hq_page: int
    hq_page_shape: Tuple[int, int]
    hq_dtype: str
    target_role: str
    is_derived: bool


@dataclass
class HQBatch:
    hq: torch.Tensor
    sample_ids: List[str]
    group_ids: List[str]
    sources: List[str]
    metadata: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PairedPageRecord:
    sample_id: str
    split: Literal["train", "val"]
    source: str
    category: str
    condition: str
    morphology: str
    group_id: str
    wf_path: Path
    wf_page: int
    wf_role: Literal["wf_lowsnr", "wf_highsnr"]
    wf_page_shape: Tuple[int, int]
    wf_dtype: str
    target_path: Path
    target_page: int
    target_role: Literal["RC_highsnr"]
    target_page_shape: Tuple[int, int]
    target_dtype: str
    spatial_scale: Literal[2]


@dataclass
class PairedBatch:
    wf: torch.Tensor
    target_hq: torch.Tensor
    wf_role: Literal["wf_lowsnr", "wf_highsnr"]
    valid_mask_hq: torch.Tensor
    sample_ids: List[str]
    group_ids: List[str]
    sources: List[str]
    lr_origins: torch.Tensor
    hr_origins: torch.Tensor
    metadata: List[Dict[str, Any]] = field(default_factory=list)
