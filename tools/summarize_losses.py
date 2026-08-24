#!/usr/bin/env python3
"""Print / CSV every generator loss: raw, weight, weighted C_i, share%.

No matplotlib. Reads metrics_train.jsonl.

  python tools/summarize_losses.py --run-dir RUN
  python tools/summarize_losses.py --run-dir RUN --last 20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

TERMS = [
    "charbonnier",
    "ms_ssim",
    "scharr",
    "hf",
    "flux",
    "dark_fp",
    "kl",
    "perceptual",
    "adv_g",
]


def _load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _f(row: Dict[str, Any], key: str) -> str:
    v = row.get(key)
    if v is None:
        return ""
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--last", type=int, default=8, help="print this many latest steps")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()
    path = args.run_dir / "metrics_train.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    rows = _load(path)
    if not rows:
        raise SystemExit("empty metrics_train.jsonl")

    csv_path = args.csv or (args.run_dir / "metrics_losses.csv")
    fields = ["step", "loss_g_total"]
    for t in TERMS:
        fields += [f"loss_raw_{t}", f"weight_{t}", f"loss_w_{t}", f"share_pct_{t}"]
    fields += ["loss_raw_disc", "loss_disc_real", "loss_disc_fake", "d_real_mean", "d_fake_mean"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            if out.get("loss_g_total") in ("", None):
                out["loss_g_total"] = row.get("loss", "")
            w.writerow(out)

    print(f"wrote {csv_path} ({len(rows)} steps)")
    print()
    show = rows[-max(args.last, 1) :]
    hdr = f"{'step':>8} {'G':>10}" + "".join(f"{t[:6]:>14}" for t in TERMS)
    print(hdr)
    print(" " * 19 + "".join(f"{'C / %':>14}" for _ in TERMS))
    for row in show:
        step = row.get("step", "")
        g = row.get("loss_g_total", row.get("loss", ""))
        line = f"{step:>8} {float(g) if g != '' else float('nan'):>10.5g}"
        for t in TERMS:
            c = row.get(f"loss_w_{t}")
            pct = row.get(f"share_pct_{t}")
            if c is None:
                line += f"{'—':>14}"
            else:
                line += f"{float(c):.3g}/{float(pct or 0):.0f}%".rjust(14)
        print(line)
    last = rows[-1]
    print()
    print(f"last step {last.get('step')}  breakdown:")
    for t in TERMS:
        raw = last.get(f"loss_raw_{t}", 0)
        wgt = last.get(f"weight_{t}", 0)
        c = last.get(f"loss_w_{t}", 0)
        pct = last.get(f"share_pct_{t}", 0)
        print(
            f"  {t:<12} raw={float(raw or 0):.6g}  "
            f"weight={float(wgt or 0):.6g}  "
            f"C={float(c or 0):.6g}  "
            f"share={float(pct or 0):.2f}%"
        )
    if last.get("d_real_mean") is not None:
        print(
            f"  {'disc':<12} Ld={last.get('loss_raw_disc')}  "
            f"real={last.get('d_real_mean')}  fake={last.get('d_fake_mean')}  "
            f"(not in G share)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
