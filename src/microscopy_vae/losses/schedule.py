"""Shared step-linear ramps used by MS-SSIM, perceptual, and adversarial weights."""

from __future__ import annotations


def scheduled_weight(
    base: float,
    step: int,
    start_step: int,
    ramp_steps: int,
) -> float:
    """Return 0 before start_step, then linear ramp over ramp_steps, then `base`.

    Matches the existing MS-SSIM schedule in HQCodecLossComposer._ms_weight:
    frac = min(1, (step - start_step + 1) / ramp_steps).
    """
    if float(base) == 0.0:
        return 0.0
    if int(step) < int(start_step):
        return 0.0
    if int(ramp_steps) <= 0:
        return float(base)
    t = int(step) - int(start_step)
    frac = min(1.0, float(t + 1) / float(ramp_steps))
    return float(base) * frac
