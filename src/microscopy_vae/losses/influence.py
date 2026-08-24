"""Three-layer loss influence: raw, weighted scalar share, periodic grad norms.

`autograd.grad` is used so `.grad` on parameters is never written. The
caller must invoke this *before* the training `backward()` and pass
`retain_graph=True` equivalents (this helper always retains the graph).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


GENERATOR_TERM_ORDER = (
    "charbonnier",
    "ms_ssim",
    "scharr",
    "hf",
    "flux",
    "dark_fp",
    "kl",
    "perceptual",
    "adv_g",
)

# Why abs() for scalar share: hinge G loss can be negative, so signed C_i / sum C_j
# can exceed 1 or go negative and is not a partition. |C_i| / sum |C_j| is a share
# of magnitude. Signed C_i remains in loss_w_*.
CONTRIB_EPS = 1e-12


def vae_param_groups(vae: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """Reference parameter sets for G_i.

    full      — every VAE parameter (needed for KL, which does not enter recon).
    encoder   — encoder + quant_conv (posterior / KL path).
    decoder   — decoder + post_quant_conv (reconstruction path).
    output    — decoder.conv_out only (pixel-term dominated).
    posterior — quant_conv only.
    """
    groups: Dict[str, List[nn.Parameter]] = {
        "full": [p for p in vae.parameters() if p.requires_grad],
        "encoder": [p for p in vae.encoder.parameters() if p.requires_grad]
        + [p for p in vae.quant_conv.parameters() if p.requires_grad],
        "decoder": [p for p in vae.decoder.parameters() if p.requires_grad]
        + [p for p in vae.post_quant_conv.parameters() if p.requires_grad],
        "output": [p for p in vae.decoder.conv_out.parameters() if p.requires_grad],
        "posterior": [p for p in vae.quant_conv.parameters() if p.requires_grad],
    }
    return groups


def _as_float(v: object) -> float:
    if v is None:
        return 0.0
    if torch.is_tensor(v):
        return float(v.detach().float().cpu())
    return float(v)


def scalar_contrib_ratios(weighted: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    abs_vals = {k: abs(_as_float(v)) for k, v in weighted.items()}
    denom = sum(abs_vals.values()) + CONTRIB_EPS
    return {k: abs_vals[k] / denom for k in weighted}


def quantify_generator_losses(
    unweighted: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    weighted: Mapping[str, torch.Tensor],
    *,
    total: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Always emit raw / weight / weighted / share for every G term (missing → 0).

    Share uses |C_i| so signed GAN terms still form a partition. Signed C_i
    stays in loss_w_*. Discriminator loss is not included.
    """
    logs: Dict[str, float] = {}
    names = list(GENERATOR_TERM_ORDER)
    for k in unweighted:
        if k not in names:
            names.append(k)
    abs_sum = 0.0
    for name in names:
        raw = _as_float(unweighted.get(name))
        w = _as_float(weights.get(name, 0.0))
        if name in weighted:
            c = _as_float(weighted[name])
        else:
            c = w * raw
        logs[f"loss_raw_{name}"] = raw
        logs[f"weight_{name}"] = w
        logs[f"loss_w_{name}"] = c
        abs_sum += abs(c)
    denom = abs_sum + CONTRIB_EPS
    for name in names:
        c = logs[f"loss_w_{name}"]
        ratio = abs(c) / denom
        logs[f"contrib_ratio_{name}"] = ratio
        logs[f"share_pct_{name}"] = 100.0 * ratio
    logs["loss_g_abs_sum"] = abs_sum
    if total is not None:
        logs["loss_g_total"] = _as_float(total)
    return logs


_TERM_SHORT = {
    "charbonnier": "char",
    "ms_ssim": "ms",
    "scharr": "scharr",
    "hf": "hf",
    "flux": "flux",
    "dark_fp": "dark",
    "kl": "kl",
    "perceptual": "perc",
    "adv_g": "adv",
}


def format_loss_breakdown(diag: Mapping[str, float]) -> str:
    """One terminal line: C_i and share% for every generator term."""
    parts = []
    gtot = diag.get("loss_g_total", diag.get("loss", float("nan")))
    parts.append(f"G={gtot:.5g}")
    for name in GENERATOR_TERM_ORDER:
        short = _TERM_SHORT.get(name, name)
        c = diag.get(f"loss_w_{name}", 0.0)
        pct = diag.get(f"share_pct_{name}", 0.0)
        parts.append(f"{short} {c:.4g} {pct:.1f}%")
    d_real = diag.get("d_real_mean")
    d_fake = diag.get("d_fake_mean")
    if d_real is not None or d_fake is not None:
        ld = diag.get("loss_raw_disc")
        parts.append(
            f"D r={'-' if d_real is None else f'{d_real:.3f}'} "
            f"f={'-' if d_fake is None else f'{d_fake:.3f}'}"
            + (f" Ld={ld:.4g}" if ld is not None else "")
        )
    return " | ".join(parts)


def _grad_l2(grads: Sequence[Optional[torch.Tensor]]) -> float:
    acc = 0.0
    for g in grads:
        if g is None:
            continue
        acc += float(g.detach().float().pow(2).sum().cpu())
    return acc ** 0.5


def _flat(grads: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    parts = [g.detach().float().reshape(-1).cpu() for g in grads if g is not None]
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def loss_grad_norms(
    terms: Mapping[str, torch.Tensor],
    params: Sequence[nn.Parameter],
) -> Dict[str, float]:
    """G_i = ||∂C_i / ∂params||_2. Does not write param.grad."""
    param_list = [p for p in params if p.requires_grad]
    if not param_list:
        return {k: 0.0 for k in terms}
    out: Dict[str, float] = {}
    for name, term in terms.items():
        if not torch.is_tensor(term) or not term.requires_grad:
            out[name] = 0.0
            continue
        if not torch.isfinite(term.detach()).all():
            out[name] = float("nan")
            continue
        grads = torch.autograd.grad(
            term,
            param_list,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        out[name] = _grad_l2(grads)
        del grads
    return out


def loss_grad_cosine(
    terms: Mapping[str, torch.Tensor],
    params: Sequence[nn.Parameter],
    pairs: Optional[Iterable[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """Cosine similarity between flattened ∂C_i and ∂C_j on `params`."""
    param_list = [p for p in params if p.requires_grad]
    names = [k for k, t in terms.items() if torch.is_tensor(t) and t.requires_grad]
    flats: Dict[str, torch.Tensor] = {}
    for name in names:
        grads = torch.autograd.grad(
            terms[name],
            param_list,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        flat = _flat(grads)
        del grads
        if flat is not None:
            flats[name] = flat
    if pairs is None:
        pair_list = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pair_list.append((a, b))
    else:
        pair_list = list(pairs)
    out: Dict[str, float] = {}
    for a, b in pair_list:
        if a not in flats or b not in flats:
            continue
        va, vb = flats[a], flats[b]
        denom = float(va.norm() * vb.norm()) + CONTRIB_EPS
        out[f"cosine_{a}_{b}"] = float((va @ vb) / denom)
    return out


def diagnose_generator_influence(
    weighted: Mapping[str, torch.Tensor],
    vae: nn.Module,
    *,
    param_group_names: Sequence[str] = ("full",),
    compute_cosine: bool = False,
) -> Dict[str, float]:
    """Periodic diagnostic. Must not be followed by optimizer.step on these grads
    because they never land in `.grad`. Training backward is still required.
    """
    groups = vae_param_groups(vae)
    logs: Dict[str, float] = {}
    active = {
        k: v
        for k, v in weighted.items()
        if torch.is_tensor(v) and v.requires_grad
    }
    for gname in param_group_names:
        if gname not in groups:
            raise KeyError(f"unknown param group {gname!r}")
        norms = loss_grad_norms(active, groups[gname])
        denom = sum(v for v in norms.values() if v == v) + CONTRIB_EPS
        for k, g in norms.items():
            logs[f"grad_norm_{gname}_{k}"] = g
            logs[f"grad_ratio_{gname}_{k}"] = (g / denom) if g == g else float("nan")
        if compute_cosine and gname in ("full", "output", "decoder"):
            for ck, cv in loss_grad_cosine(active, groups[gname]).items():
                logs[f"{gname}_{ck}"] = cv
    return logs
