"""Perceptual feature loss — microscopy migration of natural-image SR LPIPS/VGG.

Natural-image SR (SRGAN/ESRGAN): RGB in ~[0,1], ImageNet-pretrained VGG,
often after clamping. That pipeline is wrong here: S1 is 1-channel, unbounded
(can be negative), input==target codec (not LR→HR), and intensity must stay
scientifically comparable.

This module keeps the *role* (multi-scale feature L1) and changes the
adapter: frozen 1-ch conv in the global normalized domain. No RGB repeat,
no silent clamp, no per-patch min-max. Optional `vgg16` is explicit and
domain-mismatched; do not turn it on for the formal run.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_conv_stack(module: nn.Module, seed: int) -> None:
    cpu_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    finally:
        torch.set_rng_state(cpu_state)


class InternalConvExtractor(nn.Module):
    """Frozen 1-ch multi-scale conv features. No BN (inputs are unbounded)."""

    def __init__(
        self,
        *,
        channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 3,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if int(kernel_size) % 2 == 0:
            raise ValueError("perceptual kernel_size must be odd")
        chs = [1, *[int(c) for c in channels]]
        pad = int(kernel_size) // 2
        blocks = nn.ModuleDict()
        for i in range(len(chs) - 1):
            cin, cout = chs[i], chs[i + 1]
            blocks[f"block{i + 1}"] = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size, padding=pad),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(cout, cout, kernel_size, padding=pad),
                nn.LeakyReLU(0.2, inplace=False),
            )
        self.blocks = blocks
        self.block_names: List[str] = list(blocks.keys())
        _init_conv_stack(self, init_seed)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"internal_conv expects [B,1,H,W], got {tuple(x.shape)}")
        feats: Dict[str, torch.Tensor] = {}
        h = x
        for i, name in enumerate(self.block_names):
            h = self.blocks[name](h)
            feats[name] = h
            if i < len(self.block_names) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)
        return feats

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_features(x)


def freeze_extractor(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def build_internal_conv_extractor(
    *,
    channels: Sequence[int] = (16, 32, 64),
    kernel_size: int = 3,
    init_seed: int = 0,
    freeze: bool = True,
) -> InternalConvExtractor:
    ext = InternalConvExtractor(channels=channels, kernel_size=kernel_size, init_seed=init_seed)
    if freeze:
        freeze_extractor(ext)
    return ext


def adapt_vgg_input(
    x: torch.Tensor,
    *,
    repeat_channels: bool,
    clamp_to_unit: bool,
    data_mean: float,
    data_std: float,
    imagenet_mean: Sequence[float],
    imagenet_std: Sequence[float],
) -> torch.Tensor:
    """Explicit VGG adapter. Never clamps unless `clamp_to_unit=True`.

    Does **not** per-patch min-max. `data_mean`/`data_std` are fixed config
    scalars in the global normalized domain.
    """
    if x.ndim != 4:
        raise ValueError(f"VGG adapter expects [B,C,H,W], got {tuple(x.shape)}")
    y = x.float()
    if clamp_to_unit:
        y = y.clamp(0.0, 1.0)
    std = float(data_std) if float(data_std) != 0.0 else 1.0
    y = (y - float(data_mean)) / std
    if y.shape[1] == 1:
        if not repeat_channels:
            raise ValueError(
                "vgg16 requires 3-channel input; set perceptual.vgg_repeat_channels=true "
                "explicitly (this is a documented domain-mismatch, not a silent default path)"
            )
        y = y.repeat(1, 3, 1, 1)
    if y.shape[1] != 3:
        raise ValueError(f"VGG adapter produced {y.shape[1]} channels, expected 3")
    mean_t = y.new_tensor(list(imagenet_mean)).view(1, 3, 1, 1)
    std_t = y.new_tensor(list(imagenet_std)).view(1, 3, 1, 1)
    return (y - mean_t) / std_t


class VGG16Extractor(nn.Module):
    """Optional ImageNet VGG-16 feature slices. Domain mismatch is intentional."""

    _INDEX = {
        "relu1_2": 3,
        "relu2_2": 8,
        "relu3_3": 15,
        "relu4_3": 22,
    }

    def __init__(
        self,
        *,
        pretrained: bool = True,
        selected_layers: Sequence[str] = ("relu2_2", "relu3_3"),
        freeze: bool = True,
    ) -> None:
        super().__init__()
        try:
            import torchvision
            from torchvision.models import VGG16_Weights
        except ImportError as exc:  # pragma: no cover
            raise ImportError("perceptual.backbone=vgg16 requires torchvision") from exc
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = torchvision.models.vgg16(weights=weights)
        self.features = vgg.features
        wanted = list(selected_layers)
        for name in wanted:
            if name not in self._INDEX:
                raise ValueError(f"unknown VGG layer {name!r}; known={sorted(self._INDEX)}")
        self.selected_layers = wanted
        self.max_index = max(self._INDEX[n] for n in wanted)
        if freeze:
            freeze_extractor(self)

    def forward_features(self, x3: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        h = x3
        wanted = {self._INDEX[n]: n for n in self.selected_layers}
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in wanted:
                feats[wanted[i]] = h
            if i >= self.max_index:
                break
        return feats


def build_perceptual_extractor(
    *,
    backbone: str,
    freeze: bool = True,
    channels: Sequence[int] = (16, 32, 64),
    kernel_size: int = 3,
    init_seed: int = 0,
    selected_layers: Optional[Sequence[str]] = None,
    vgg_pretrained: bool = True,
) -> nn.Module:
    if backbone == "internal_conv":
        return build_internal_conv_extractor(
            channels=channels, kernel_size=kernel_size, init_seed=init_seed, freeze=freeze
        )
    if backbone == "vgg16":
        layers = selected_layers or ("relu2_2", "relu3_3")
        return VGG16Extractor(pretrained=vgg_pretrained, selected_layers=layers, freeze=freeze)
    raise ValueError(f"unknown perceptual backbone {backbone!r}")


def _maybe_unit_features(feat: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return feat
    # Channel-wise unit variance over spatial dims; no per-patch min-max on pixels.
    var = feat.var(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-8)
    return feat / var.sqrt()


def perceptual_feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    extractor: nn.Module,
    *,
    selected_layers: Sequence[str],
    layer_weights: Optional[Dict[str, float]] = None,
    distance: str = "l1",
    normalize_features: bool = False,
    unstructured_policy: str = "keep",
    support: Optional[torch.Tensor] = None,
    vgg_adapter_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Feature distance. `pred`/`target` already in the chosen intensity domain.

    `unstructured_policy='match_target'` copies target onto unstructured pixels
    of *both* tensors' unused locations (pred only; target is already target).
    This is the same idea as the MS-SSIM gate: do not teach the extractor to
    match background spikes. It is not a clamp and does not zero negatives.
    """
    pred_f = pred.float()
    tgt_f = target.float()
    diag: Dict[str, torch.Tensor] = {
        "perc_input_min": pred_f.amin().detach(),
        "perc_input_max": pred_f.amax().detach(),
        "perc_target_min": tgt_f.amin().detach(),
        "perc_target_max": tgt_f.amax().detach(),
        "perc_clamped": pred_f.new_zeros(()),
    }
    if unstructured_policy == "match_target" and support is not None:
        m = support.to(dtype=pred_f.dtype)
        pred_f = torch.where(m > 0, pred_f, tgt_f)
    elif unstructured_policy not in ("keep", "match_target"):
        raise ValueError(f"unknown unstructured_policy={unstructured_policy!r}")

    pred_in, tgt_in = pred_f, tgt_f
    if vgg_adapter_kwargs is not None:
        if bool(vgg_adapter_kwargs.get("clamp_to_unit", False)):
            diag["perc_clamped"] = pred_f.new_ones(())
        pred_in = adapt_vgg_input(pred_f, **vgg_adapter_kwargs)
        with torch.no_grad():
            tgt_in = adapt_vgg_input(tgt_f, **vgg_adapter_kwargs)

    extractor.eval()
    pred_feats = extractor.forward_features(pred_in)
    with torch.no_grad():
        tgt_feats = extractor.forward_features(tgt_in)

    if not selected_layers:
        raise ValueError("perceptual selected_layers is empty")
    acc = pred_f.new_zeros(())
    wsum = 0.0
    for name in selected_layers:
        if name not in pred_feats:
            raise KeyError(f"extractor has no layer {name!r}; have {sorted(pred_feats)}")
        w = 1.0 if not layer_weights else float(layer_weights.get(name, 1.0))
        a = _maybe_unit_features(pred_feats[name], normalize_features)
        b = _maybe_unit_features(tgt_feats[name], normalize_features)
        if distance == "l1":
            d = (a - b).abs().mean()
        elif distance == "l2":
            d = (a - b).pow(2).mean()
        else:
            raise ValueError(f"unknown perceptual distance {distance!r}")
        acc = acc + w * d
        wsum += abs(w)
        diag[f"perc_layer_{name}"] = d.detach()
    if wsum <= 0:
        raise ValueError("perceptual layer_weights sum to 0")
    # Mean over listed layers so `weight` in yaml is not secretly scaled by n_layers.
    loss = acc / float(len(selected_layers))
    return loss, diag
