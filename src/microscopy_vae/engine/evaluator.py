"""HQ codec evaluation: posterior-mean recon + group-macro aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from microscopy_vae.data.records import HQBatch
from microscopy_vae.metrics.aggregation import (
    group_bootstrap_ci,
    page_to_group_macro,
    volume_mse_pooled_psnr,
)
from microscopy_vae.metrics.extended import nmse, ssim_mean
from microscopy_vae.metrics.fidelity import mae, mse, psnr, signed_mean_bias
from microscopy_vae.systems.hq_codec import HQCodecSystem


@torch.no_grad()
def evaluate_hq_loader(
    system: HQCodecSystem,
    loader: DataLoader,
    *,
    device: torch.device,
    use_posterior_mean: bool = True,
    max_batches: Optional[int] = None,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 0,
    report_constant_baseline: bool = True,
    extended_metrics: bool = False,
) -> Dict[str, Any]:
    system.eval()
    page_metrics: List[Dict[str, float]] = []
    group_ids: List[str] = []
    sources: List[str] = []
    const_page: List[Dict[str, float]] = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        assert isinstance(batch, HQBatch)
        x = batch.hq.to(device, non_blocking=True)
        if use_posterior_mean:
            recon = system.reconstruct_hq(x)
            post, _ = system.encode_hq(x, sample_posterior=False)
        else:
            out = system.vae(x, sample_posterior=True)
            recon = out.reconstruction
            post = out.posterior
        for i in range(x.shape[0]):
            xi = x[i : i + 1]
            ri = recon[i : i + 1]
            xi_np = xi.detach().float().cpu().numpy()[0, 0]
            ri_np = ri.detach().float().cpu().numpy()[0, 0]
            m = {
                "mae": float(mae(ri, xi).cpu()),
                "mse": float(mse(ri, xi).cpu()),
                "psnr": float(psnr(ri, xi).cpu()),
                "signed_bias": float(signed_mean_bias(ri, xi).cpu()),
                "abs_bias": float((ri.mean() - xi.mean()).abs().cpu()),
                "kl_mean": float(
                    0.5
                    * (post.mean[i].pow(2) + post.var[i] - 1.0 - post.logvar[i]).mean().cpu()
                ),
            }
            if extended_metrics:
                m["nmse"] = float(nmse(ri_np, xi_np))
                snr = -10.0 * np.log10(m["nmse"]) if m["nmse"] > 0 and np.isfinite(m["nmse"]) else float("nan")
                m["snr_db"] = float(snr)
                m["ssim_range1"] = float(ssim_mean(ri_np, xi_np, data_range=1.0))
                m["target_std"] = float(xi_np.std())
            page_metrics.append(m)
            group_ids.append(batch.group_ids[i])
            sources.append(batch.sources[i])
            if report_constant_baseline:
                c = torch.full_like(xi, float(xi.mean().cpu()))
                const_page.append(
                    {
                        "mae": float(mae(c, xi).cpu()),
                        "mse": float(mse(c, xi).cpu()),
                        "psnr": float(psnr(c, xi).cpu()),
                        "signed_bias": float(signed_mean_bias(c, xi).cpu()),
                        "abs_bias": 0.0,
                        "kl_mean": 0.0,
                    }
                )

    keys = ["mae", "mse", "psnr", "signed_bias", "abs_bias", "kl_mean"]
    if extended_metrics:
        keys.extend(["nmse", "snr_db", "ssim_range1", "target_std"])
    macro = page_to_group_macro(page_metrics, group_ids, keys)
    macro["psnr_mse_pooled"] = volume_mse_pooled_psnr(
        [m["mse"] for m in page_metrics], group_ids, data_range=1.0
    )

    by_source: Dict[str, Dict[str, float]] = {}
    source_set = sorted(set(sources))
    for s in source_set:
        idx = [i for i, ss in enumerate(sources) if ss == s]
        by_source[s] = page_to_group_macro(
            [page_metrics[i] for i in idx],
            [group_ids[i] for i in idx],
            keys,
        )
    # equal-source macro: mean of per-source macros
    equal_source_macro = {}
    if by_source:
        for k in keys:
            vals = [by_source[s][k] for s in by_source if k in by_source[s]]
            equal_source_macro[k] = float(sum(vals) / max(len(vals), 1))

    group_psnr: Dict[str, List[float]] = defaultdict(list)
    for m, g in zip(page_metrics, group_ids):
        group_psnr[g].append(m["psnr"])
    group_vals = [float(sum(v) / len(v)) for v in group_psnr.values()]
    boot = group_bootstrap_ci(group_vals, n_resamples=bootstrap_n, seed=bootstrap_seed)

    out: Dict[str, Any] = {
        "n_pages": len(page_metrics),
        "n_groups": len(set(group_ids)),
        "group_macro": macro,
        "by_source": by_source,
        "equal_source_macro": equal_source_macro,
        "psnr_bootstrap": boot,
        "page_metrics": page_metrics,
        "group_ids": group_ids,
    }
    if report_constant_baseline and const_page:
        const_keys = [k for k in keys if k in const_page[0]]
        out["constant_baseline"] = {
            "group_macro": page_to_group_macro(const_page, group_ids, const_keys),
            "note": "per-image constant = mean(target); high PSNR means low variance not good recon",
        }
    return out
