"""HQ codec evaluation: posterior-mean recon + group-macro aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from microscopy_vae.data.records import HQBatch
from microscopy_vae.engine.distributed import DistInfo, all_gather_objects, raise_if_any_rank_failed
from microscopy_vae.metrics.aggregation import (
    group_bootstrap_ci,
    page_to_group_macro,
    volume_mse_pooled_psnr,
)
from microscopy_vae.metrics.extended import (
    background_false_positive_stats,
    nmse,
    ssim_local,
    ssim_mean,
    target_robust_range,
)
from microscopy_vae.metrics.fidelity import mae, mse, psnr, signed_mean_bias
from microscopy_vae.systems.hq_codec import HQCodecSystem

_BASE_KEYS = (
    "mae",
    "mse",
    "psnr",
    "signed_bias",
    "abs_bias",
    "kl_mean",
    "nmse",
    "ssim_local",
    "target_robust_range",
)
_EXT_KEYS = ("snr_db", "ssim_range1", "target_std", "bg_fp_rate", "bg_bias", "bg_mae")


def _page_row(
    ri: torch.Tensor,
    xi: torch.Tensor,
    *,
    post_mean: torch.Tensor,
    post_var: torch.Tensor,
    post_logvar: torch.Tensor,
    extended_metrics: bool,
) -> Dict[str, float]:
    xi_np = xi.detach().float().cpu().numpy()[0, 0]
    ri_np = ri.detach().float().cpu().numpy()[0, 0]
    m: Dict[str, float] = {
        "mae": float(mae(ri, xi).cpu()),
        "mse": float(mse(ri, xi).cpu()),
        "psnr": float(psnr(ri, xi).cpu()),
        "signed_bias": float(signed_mean_bias(ri, xi).cpu()),
        "abs_bias": float((ri.mean() - xi.mean()).abs().cpu()),
        "kl_mean": float(0.5 * (post_mean.pow(2) + post_var - 1.0 - post_logvar).mean().cpu()),
        "nmse": float(nmse(ri_np, xi_np)),
        "ssim_local": float(ssim_local(ri_np, xi_np)),
        "target_robust_range": float(target_robust_range(xi_np)),
    }
    if extended_metrics:
        snr = -10.0 * np.log10(m["nmse"]) if m["nmse"] > 0 and np.isfinite(m["nmse"]) else float("nan")
        m["snr_db"] = float(snr)
        m["ssim_range1"] = float(ssim_mean(ri_np, xi_np, data_range=1.0))
        m["target_std"] = float(xi_np.std())
        m.update(background_false_positive_stats(ri_np, xi_np))
    return m


def _const_row(xi: torch.Tensor, *, extended_metrics: bool) -> Dict[str, float]:
    mean_v = float(xi.mean().cpu())
    c = torch.full_like(xi, mean_v)
    xi_np = xi.detach().float().cpu().numpy()[0, 0]
    c_np = np.full(xi_np.shape, mean_v, dtype=np.float32)
    m: Dict[str, float] = {
        "mae": float(mae(c, xi).cpu()),
        "mse": float(mse(c, xi).cpu()),
        "psnr": float(psnr(c, xi).cpu()),
        "signed_bias": float(signed_mean_bias(c, xi).cpu()),
        "abs_bias": 0.0,
        "kl_mean": 0.0,
        "nmse": float(nmse(c_np, xi_np)),
        "ssim_local": float(ssim_local(c_np, xi_np)),
        "target_robust_range": float(target_robust_range(xi_np)),
    }
    if extended_metrics:
        snr = -10.0 * np.log10(m["nmse"]) if m["nmse"] > 0 and np.isfinite(m["nmse"]) else float("nan")
        m["snr_db"] = float(snr)
        m["ssim_range1"] = float(ssim_mean(c_np, xi_np, data_range=1.0))
        m["target_std"] = float(xi_np.std())
        m.update(background_false_positive_stats(c_np, xi_np))
    return m


def aggregate_eval_pages(
    page_metrics: Sequence[Dict[str, float]],
    group_ids: Sequence[str],
    sources: Sequence[str],
    *,
    bootstrap_n: int,
    bootstrap_seed: int,
    extended_metrics: bool,
    const_page: Optional[Sequence[Dict[str, float]]] = None,
    report_constant_baseline: bool = True,
    sample_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Group-macro / by-source / bootstrap. Order of ``page_metrics`` must match
    the single-process dataset order (sort by dataset_index before calling)."""
    keys = list(_BASE_KEYS)
    if extended_metrics:
        keys.extend(_EXT_KEYS)
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
    equal_source_macro: Dict[str, float] = {}
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
        "page_metrics": list(page_metrics),
        "group_ids": list(group_ids),
        "sources": list(sources),
    }
    if sample_ids is not None:
        out["sample_ids"] = list(sample_ids)
    if report_constant_baseline and const_page:
        const_keys = [k for k in keys if const_page and k in const_page[0]]
        out["constant_baseline"] = {
            "group_macro": page_to_group_macro(const_page, group_ids, const_keys),
            "note": "per-image constant = mean(target); high PSNR means low variance not good recon",
        }
    return out


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
    dist: Optional[DistInfo] = None,
    expected_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-page metrics then the same group-macro as the single-process path.

    One encoder+decoder forward per batch (recon and KL share the posterior).
    When ``dist`` is enabled, each rank computes a disjoint shard and records
    are gathered, sorted by ``dataset_index``, then aggregated on every rank
    (same dict). Do not pad/repeat samples.
    """
    system.eval()
    records: List[Dict[str, Any]] = []
    local_i = 0
    err = ""
    try:
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            assert isinstance(batch, HQBatch)
            x = batch.hq.to(device, non_blocking=True)
            out = system.vae(x, sample_posterior=not bool(use_posterior_mean))
            recon = out.reconstruction
            post = out.posterior
            for i in range(x.shape[0]):
                xi = x[i : i + 1]
                ri = recon[i : i + 1]
                meta = batch.metadata[i] if batch.metadata else {}
                ds_idx = meta.get("dataset_index")
                if ds_idx is None:
                    ds_idx = local_i
                row = _page_row(
                    ri,
                    xi,
                    post_mean=post.mean[i],
                    post_var=post.var[i],
                    post_logvar=post.logvar[i],
                    extended_metrics=extended_metrics,
                )
                rec: Dict[str, Any] = {
                    "dataset_index": int(ds_idx),
                    "sample_id": str(batch.sample_ids[i]),
                    "group_id": str(batch.group_ids[i]),
                    "source": str(batch.sources[i]),
                    "metrics": row,
                }
                if report_constant_baseline:
                    rec["const"] = _const_row(xi, extended_metrics=extended_metrics)
                records.append(rec)
                local_i += 1
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    if dist is not None:
        raise_if_any_rank_failed(not err, err or "evaluate_hq_loader failed", dist)
    elif err:
        raise RuntimeError(err)

    if dist is not None and dist.enabled:
        gathered = all_gather_objects(records, dist)
        merged: List[Dict[str, Any]] = []
        for chunk in gathered:
            merged.extend(chunk or [])
        records = merged

    if records and all("dataset_index" in r for r in records):
        records.sort(key=lambda r: (int(r["dataset_index"]), str(r["sample_id"])))
        idxs = [int(r["dataset_index"]) for r in records]
        if len(idxs) != len(set(idxs)):
            raise RuntimeError(
                f"val shard produced duplicate dataset_index values (n={len(idxs)} unique={len(set(idxs))})"
            )
        if expected_n is not None and max_batches is None:
            want = set(range(int(expected_n)))
            got = set(idxs)
            if got != want:
                raise RuntimeError(
                    f"val coverage mismatch: missing={sorted(want - got)[:12]} "
                    f"extra={sorted(got - want)[:12]} expected_n={expected_n}"
                )

    page_metrics = [r["metrics"] for r in records]
    group_ids = [r["group_id"] for r in records]
    sources = [r["source"] for r in records]
    sample_ids = [r["sample_id"] for r in records]
    const_page = [r["const"] for r in records if "const" in r] if report_constant_baseline else None
    return aggregate_eval_pages(
        page_metrics,
        group_ids,
        sources,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
        extended_metrics=extended_metrics,
        const_page=const_page,
        report_constant_baseline=report_constant_baseline,
        sample_ids=sample_ids,
    )
