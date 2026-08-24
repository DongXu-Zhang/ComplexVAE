from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    data_range: float = 1.0,
    pad_mode: str = "reflect",
    reduce: bool = True,
) -> torch.Tensor:
    """1 - SSIM (single scale), FP32. Default reflect pad (not zero) to avoid crop-edge artifacts."""
    pred_f = pred.float()
    target_f = target.float()
    channel = pred_f.shape[1]
    window = _gaussian_window(window_size, device=pred_f.device, dtype=pred_f.dtype)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    if pad_mode == "reflect":
        pred_f = F.pad(pred_f, (pad, pad, pad, pad), mode="reflect")
        target_f = F.pad(target_f, (pad, pad, pad, pad), mode="reflect")
        conv_pad = 0
    else:
        conv_pad = pad
    mu_x = F.conv2d(pred_f, window, padding=conv_pad, groups=channel)
    mu_y = F.conv2d(target_f, window, padding=conv_pad, groups=channel)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(pred_f * pred_f, window, padding=conv_pad, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(target_f * target_f, window, padding=conv_pad, groups=channel) - mu_y2
    sigma_xy = F.conv2d(pred_f * target_f, window, padding=conv_pad, groups=channel) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    per = 1.0 - ssim_map.mean(dim=(1, 2, 3))
    if reduce:
        return per.mean()
    return per


def ms_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    levels: int = 4,
    data_range: float = 1.0,
    pad_mode: str = "reflect",
    reduce: bool = True,
) -> torch.Tensor:
    """Simplified multi-scale SSIM: weighted average of (1-SSIM) over pyramid levels.

    Not the multiplicative canonical MS-SSIM product form; documented as simplified.
    """
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
    weights = [w / sum(weights) for w in weights]
    total = pred.new_zeros(pred.shape[0]) if not reduce else pred.new_zeros(())
    x, y = pred, target
    for i, w in enumerate(weights):
        total = total + w * ssim_loss(
            x, y, data_range=data_range, pad_mode=pad_mode, reduce=reduce
        )
        if i < levels - 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, stride=2, ceil_mode=True)
    return total


# Scharr kernels (3x3), L1-normalized so magnitude is comparable to image units
_SCHARR_X = torch.tensor(
    [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], dtype=torch.float32
)
_SCHARR_Y = torch.tensor(
    [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]], dtype=torch.float32
)
_SCHARR_X = _SCHARR_X / _SCHARR_X.abs().sum()
_SCHARR_Y = _SCHARR_Y / _SCHARR_Y.abs().sum()


def _scharr_kernels(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = x.shape[1]
    kx = _SCHARR_X.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    ky = _SCHARR_Y.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 3).expand(c, 1, 3, 3)
    return kx, ky


def scharr_components(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflect-padded Scharr gx, gy, same spatial shape as x. Always FP32.

    Kernels must match the padded tensor dtype. Building them from the original
    x (possibly bf16 under autocast) makes conv2d see float input + bf16 weight.
    """
    x_f = F.pad(x.float(), (1, 1, 1, 1), mode="reflect")
    kx, ky = _scharr_kernels(x_f)
    c = x_f.shape[1]
    return F.conv2d(x_f, kx, padding=0, groups=c), F.conv2d(x_f, ky, padding=0, groups=c)


def scharr_magnitude(x: torch.Tensor) -> torch.Tensor:
    """|gx| + |gy|, shape [B,C,H,W]."""
    gx, gy = scharr_components(x)
    return gx.abs() + gy.abs()


def scharr_grad_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel |Δgx| + |Δgy|, shape [B,C,H,W]."""
    gx_p, gy_p = scharr_components(pred)
    gx_t, gy_t = scharr_components(target)
    return (gx_p - gx_t).abs() + (gy_p - gy_t).abs()


def scharr_grad_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reduce: bool = True,
) -> torch.Tensor:
    """Mean absolute Scharr gradient error; reflect pad + L1-normalized kernels."""
    per_map = scharr_grad_map(pred, target)
    per = per_map.mean(dim=(1, 2, 3))
    if reduce:
        return per.mean()
    return per
