"""EX-1 (One-to-One): pre-training SSL di dataset S, lalu fine-tuning + evaluasi di S.

    python scripts/run_ex1.py --source RE1 --model ts2vec --seeds 0 1 2

Untuk setiap seed:
  1. Pre-training TS2Vec pada window per service dari kasus split TRAIN (seluruh rekaman, tanpa label).
  2. Fine-tuning lokalisasi root service: FineLast dan FineAll.
  3. Kontrol "scratch": encoder acak (tanpa pre-training) dengan FineLast/FineAll yang sama.
  4. Evaluasi di split TEST; baseline heuristik dihitung di split test yang sama.
Hasil: results/ex1/<source>_<model>/ (predictions.csv, summary.csv, history.csv, pretrain_loss.csv)
       checkpoints/ex1/<source>_<model>_seed<s>.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_rcl.baseline import heuristic_table  # noqa: E402
from ssl_rcl.data import CORE_METRICS, build_samples, make_pretrain_windows, stratified_split  # noqa: E402
from ssl_rcl.finetune import predict_table, summarize_ranks, train_fineall, train_finelast  # noqa: E402
from ssl_rcl.models.timemae import TimeMAEEncoder, pretrain_timemae  # noqa: E402
from ssl_rcl.models.ts2vec import TSEncoder, pretrain_ts2vec  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--source", default="RE1")
p.add_argument("--model", default="ts2vec", choices=["ts2vec", "timemae"])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--n-iters", type=int, default=None,
               help="iterasi pre-training (default: ts2vec 600 = kode resmi, timemae 3000)")
p.add_argument("--pt-window", type=int, default=540)
p.add_argument("--pt-stride", type=int, default=270)
p.add_argument("--fineall-epochs", type=int, default=30)
p.add_argument("--no-scratch", action="store_true", help="lewati kontrol encoder acak")
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
               ("cuda" if torch.cuda.is_available() else "cpu"))
p.add_argument("--pretrain-device", default="auto",
               help="auto = CPU untuk ts2vec di MPS (random crop TS2Vec membuat panjang input berubah tiap "
                    "iterasi, dan MPS meng-cache graph per bentuk input sehingga memori membengkak); "
                    "selain itu sama dengan --device.")
# --- hiperparameter TimeMAE (default mengikuti paper; patch 12 membagi inject_offset 300 dan T 540)
p.add_argument("--tm-patch", type=int, default=12)
p.add_argument("--tm-dim", type=int, default=64)
p.add_argument("--tm-depth", type=int, default=8)
p.add_argument("--tm-vocab", type=int, default=128)
p.add_argument("--tm-mask-ratio", type=float, default=0.6)
p.add_argument("--tm-batch", type=int, default=64)
p.add_argument("--tm-gamma", type=float, default=5.0, help="bobot loss keragaman codeword (bukan dari paper)")
args = p.parse_args()
if args.n_iters is None:
    args.n_iters = 600 if args.model == "ts2vec" else 3000
if args.pretrain_device == "auto":
    args.pretrain_device = "cpu" if (args.model == "ts2vec" and args.device == "mps") else args.device

device = torch.device(args.device)
pt_device = torch.device(args.pretrain_device)
out_dir = ROOT / "results" / "ex1" / f"{args.source}_{args.model}"
ck_dir = ROOT / "checkpoints" / "ex1"
out_dir.mkdir(parents=True, exist_ok=True); ck_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

K = len(CORE_METRICS)
cache = ROOT / "cache"
ft_samples = build_samples(ROOT / "dataset", [args.source], cache_dir=cache, verbose=False)
# Seluruh rekaman (untuk pre-training) tidak di-cache ke disk; cepat dihitung ulang.
full_samples = build_samples(ROOT / "dataset", [args.source], pre=None, post=None, cache_dir=None, verbose=False)
tr_idx, va_idx, te_idx = stratified_split(ft_samples, seed=42)       # split tetap, sama untuk semua seed
train = [ft_samples[i] for i in tr_idx]; val = [ft_samples[i] for i in va_idx]; test = [ft_samples[i] for i in te_idx]
train_names = {s["name"] for s in train}
pt_windows = make_pretrain_windows([s for s in full_samples if s["name"] in train_names],
                                   args.pt_window, args.pt_stride, with_mask=True)
del full_samples                                                       # hemat memori
print(f"{args.source}: train/val/test = {len(train)}/{len(val)}/{len(test)} | window pre-training {pt_windows.shape}"
      f" | device {device}")

pred_rows, summary_rows, hist_rows, pt_logs = [], [], [], []


def pretrain(windows, dev, seed):
    if args.model == "ts2vec":
        return pretrain_ts2vec(windows, 2 * K, dev, n_iters=args.n_iters, seed=seed)
    return pretrain_timemae(windows, 2 * K, dev, n_iters=args.n_iters, batch_size=args.tm_batch,
                            output_dims=args.tm_dim, patch_size=args.tm_patch, depth=args.tm_depth,
                            vocab_size=args.tm_vocab, mask_ratio=args.tm_mask_ratio, gamma=args.tm_gamma, seed=seed)


def new_encoder():
    """Encoder acak (kontrol scratch) dengan arsitektur yang sama seperti hasil pre-training."""
    if args.model == "ts2vec":
        return TSEncoder(2 * K)
    return TimeMAEEncoder(2 * K, args.tm_dim, args.tm_patch, args.tm_depth)


def record(tag_variant, mode, seed, result, secs):
    pt = predict_table(result["encoder"], result["head"], test, device)
    pt["variant"], pt["mode"], pt["seed"] = tag_variant, mode, seed
    pred_rows.append(pt)
    summary_rows.append(dict(variant=tag_variant, mode=mode, seed=seed, train_sec=round(secs, 1),
                             epochs=len(result["history"]), **summarize_ranks(pt["rank"])))
    h = result["history"].copy(); h["variant"], h["mode"], h["seed"] = tag_variant, mode, seed
    hist_rows.append(h)
    print(f"  {tag_variant:9s} {mode:8s} seed {seed}: " +
          " ".join(f"{k}={v:.3f}" for k, v in summarize_ranks(pt["rank"]).items() if k != "n") + f" ({secs:.0f}s)")


for seed in args.seeds:
    print(f"\n=== seed {seed} ===")
    t = time.time()
    enc, log = pretrain(pt_windows, pt_device, seed)
    enc = enc.to(device)
    print(f"  pre-training selesai ({time.time() - t:.0f}s)")
    torch.save(enc.state_dict(), ck_dir / f"{args.source}_{args.model}_seed{seed}.pt")
    pt_logs.append(pd.DataFrame(dict(seed=seed, iter=np.arange(1, len(log) + 1), loss=log)))

    torch.manual_seed(seed)
    encoders = {"pretrain": enc}
    if not args.no_scratch:
        encoders["scratch"] = new_encoder().to(device).eval()

    for variant, e in encoders.items():
        t = time.time(); r = train_finelast(e, train, val, device, seed=seed); record(variant, "FineLast", seed, r, time.time() - t)
        t = time.time(); r = train_fineall(e, train, val, device, epochs=args.fineall_epochs, seed=seed)
        record(variant, "FineAll", seed, r, time.time() - t)

    # simpan bertahap supaya hasil tidak hilang bila proses terhenti
    pd.concat(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    pd.concat(hist_rows).to_csv(out_dir / "history.csv", index=False)
    pd.concat(pt_logs).to_csv(out_dir / "pretrain_loss.csv", index=False)

heur = heuristic_table(test); heur["variant"], heur["mode"], heur["seed"] = "heuristic", "-", -1
heur.to_csv(out_dir / "heuristic_test.csv", index=False)

summary = pd.DataFrame(summary_rows)
agg = summary.groupby(["variant", "mode"])[["AC@1", "AC@3", "AC@5", "Avg@5"]].agg(["mean", "std"]).round(3)
print("\n== Ringkasan test (mean ± std antar seed) ==")
print(agg.to_string())
print("\nHeuristik (test):", {k: round(v, 3) for k, v in summarize_ranks(heur["rank"]).items()})
