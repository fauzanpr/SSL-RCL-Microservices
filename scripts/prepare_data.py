"""Bangun cache praproses dan cek kewajaran data.

    python scripts/prepare_data.py --re RE1 RE2 RE3
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_rcl.data import (CORE_METRICS, EXTENDED_METRICS, build_samples, case_meta_table,  # noqa: E402
                          collate_cases, make_pretrain_windows, stratified_split)

p = argparse.ArgumentParser()
p.add_argument("--re", nargs="+", default=["RE1", "RE2", "RE3"])
p.add_argument("--extended", action="store_true", help="K=7 (tambah diskio, socket)")
p.add_argument("--pre", type=int, default=300)
p.add_argument("--post", type=int, default=240)
args = p.parse_args()

metrics = EXTENDED_METRICS if args.extended else CORE_METRICS
samples = build_samples(ROOT / "dataset", args.re, metrics, args.pre, args.post, cache_dir=ROOT / "cache")
meta = case_meta_table(samples)

print("\n== Ringkasan ==")
print(meta.groupby(["re", "system"]).agg(n=("name", "size"), n_svc=("n_services", "unique"),
                                         T=("T", "unique"), n_root=("root_service", "nunique"),
                                         masked=("masked_channels", "mean")).to_string())

for re_name in args.re:
    sub = [s for s in samples if s["re"] == re_name]
    tr, va, te = stratified_split(sub)
    print(f"\n{re_name} split train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    w = make_pretrain_windows([sub[i] for i in tr], length=args.pre + args.post, stride=args.pre + args.post)
    print(f"  window pre-training (train saja): {w.shape}, nilai min/max {w.min():.2f}/{w.max():.2f}")

# Sanity: tidak ada NaN/inf, label valid, baseline kanal aktif ~ terpusat.
allx = np.concatenate([s["x"].reshape(-1, s["x"].shape[-1]) for s in samples])
assert np.isfinite(allx).all(), "ada NaN/inf"
assert all(s["services"][s["label"]] == s["root_service"] for s in samples)
batch = collate_cases(samples[:4])
print("\nbatch:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})

# Sinyal di root service: |z| rata-rata pasca-injeksi pada root vs service lain.
root_dev, other_dev = [], []
for s in samples:
    post = np.abs(s["x"][:, s["inject_offset"]:])        # [N, T_post, K]
    m = s["mask"][:, None, :]
    # rata-rata atas kanal aktif saja (membagi dengan K penuh menguntungkan service berkanal banyak)
    dev = (post * m).sum(axis=(1, 2)) / np.maximum(m.sum(axis=(1, 2)) * post.shape[1], 1)
    root_dev.append(dev[s["label"]]); other_dev.append(np.delete(dev, s["label"]).mean())
    s["_rank"] = int((dev > dev[s["label"]]).sum()) + 1
meta["rank_by_deviation"] = [s["_rank"] for s in samples]
print(f"\n|z| pasca-injeksi rata-rata: root={np.mean(root_dev):.2f}, service lain={np.mean(other_dev):.2f}")
print("\nBaseline heuristik (ranking service berdasarkan |z| pasca-injeksi), AC@1 / AC@3:")
print(meta.groupby(["re", "system"]).rank_by_deviation.agg(
    AC1=lambda r: (r <= 1).mean(), AC3=lambda r: (r <= 3).mean()).round(3).to_string())
