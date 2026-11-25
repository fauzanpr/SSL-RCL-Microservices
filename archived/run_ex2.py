"""EX-2 (One-to-Many / transfer): encoder hasil pre-training di dataset sumber S, fine-tuning + evaluasi di target T.

    python scripts/run_ex2.py --target RE2                              # sumber default: RE lain
    python scripts/run_ex2.py --target RE2 --modes FineLast --fractions 0.1 0.25 0.5 1.0
    python scripts/run_ex2.py --target RE1 --sources RE2 --seeds 0 --resume

Script berjalan per TARGET supaya semua varian dibandingkan pada split, subset label, dan seed yang sama:
  - transfer : encoder checkpoints/ex1/<S>_<model>_seed<s>.pt (pre-training di S), S != T
  - indomain : encoder checkpoints/ex1/<T>_<model>_seed<s>.pt (setting EX-1, sebagai acuan atas)
  - scratch  : encoder acak (kontrol, tanpa pre-training)
Encoder sumber diambil dari EX-1 (pre-training di split train S saja), jadi tidak ada pre-training ulang.

Fine-tuning memakai split train target (stratified_split seed 42). Dengan --fractions < 1, split train
di-subsample terstratifikasi per (sistem, root service), minimal 1 kasus per grup. Subset bergantung pada
seed dan sama untuk semua varian dalam seed tersebut. Split val dan test selalu penuh.

Hasil: results/ex2/<T>_<model>/ (summary.csv, predictions.csv, history.csv, heuristic_test.csv, config.json)
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
from ssl_rcl.data import CORE_METRICS, build_samples, stratified_split  # noqa: E402
from ssl_rcl.finetune import predict_table, summarize_ranks, train_fineall, train_finelast  # noqa: E402
from ssl_rcl.models.run import RUNEncoder  # noqa: E402
from ssl_rcl.models.timemae import TimeMAEEncoder  # noqa: E402
from ssl_rcl.models.ts2vec import TSEncoder  # noqa: E402

ALL_RE = ["RE1", "RE2", "RE3"]
ENCODERS = {"ts2vec": TSEncoder, "timemae": TimeMAEEncoder, "run": RUNEncoder}

p = argparse.ArgumentParser()
p.add_argument("--target", required=True, choices=ALL_RE)
p.add_argument("--sources", nargs="+", default=None, choices=ALL_RE, help="default: semua RE selain target")
p.add_argument("--model", default="ts2vec", choices=list(ENCODERS))
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--fractions", nargs="+", type=float, default=[1.0], help="fraksi split train target yang dipakai")
p.add_argument("--modes", nargs="+", default=["FineLast", "FineAll"], choices=["FineLast", "FineAll"])
p.add_argument("--no-indomain", action="store_true", help="lewati varian indomain")
p.add_argument("--no-scratch", action="store_true", help="lewati varian scratch")
p.add_argument("--fineall-epochs", type=int, default=30)
p.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "ex1")
p.add_argument("--out-dir", type=Path, default=None, help="default: results/ex2/<target>_<model>")
p.add_argument("--resume", action="store_true", help="lewati kombinasi yang sudah ada di summary.csv")
p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
               ("cuda" if torch.cuda.is_available() else "cpu"))
args = p.parse_args()

sources = args.sources or [r for r in ALL_RE if r != args.target]
if args.target in sources:
    p.error("target tidak boleh ada di --sources (itu varian indomain)")
if any(not 0 < f <= 1 for f in args.fractions):
    p.error("--fractions harus di (0, 1]")

device = torch.device(args.device)
K = len(CORE_METRICS)


def encoder_for(re_name):
    """Encoder kosong dengan arsitektur yang sama seperti saat pre-training di `re_name` (EX-1).

    TimeMAE/RUN: hiperparameter arsitektur dibaca dari results/ex1/<RE>_<model>/config.json, karena checkpoint
    hanya berisi bobot. Tanpa itu, `--tm-*`/`--run-*` yang non-default akan gagal dimuat.
    """
    if args.model == "ts2vec":
        return TSEncoder(2 * K)
    cfg_path = ROOT / "results" / "ex1" / f"{re_name}_{args.model}" / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if args.model == "run":
        return RUNEncoder(2 * K, cfg.get("run_dim", 64), cfg.get("run_window", 32), cfg.get("run_ma", 25),
                          cfg.get("run_mask_p", 0.5))
    return TimeMAEEncoder(2 * K, cfg.get("tm_dim", 64), cfg.get("tm_patch", 12), cfg.get("tm_depth", 8))

# Semua checkpoint dicek di awal supaya run panjang tidak gagal di tengah.
needed = [(s, sd) for s in sources + ([] if args.no_indomain else [args.target]) for sd in args.seeds]
missing = [args.ckpt_dir / f"{s}_{args.model}_seed{sd}.pt" for s, sd in needed
           if not (args.ckpt_dir / f"{s}_{args.model}_seed{sd}.pt").exists()]
if missing:
    sys.exit("Checkpoint tidak ditemukan:\n  " + "\n  ".join(map(str, missing)))

out_dir = args.out_dir or ROOT / "results" / "ex2" / f"{args.target}_{args.model}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(
    {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"sources": sources}, indent=2))

samples = build_samples(ROOT / "dataset", [args.target], cache_dir=ROOT / "cache", verbose=False)
tr_idx, va_idx, te_idx = stratified_split(samples, seed=42)            # split tetap, sama seperti EX-1
train_full = [samples[i] for i in tr_idx]
val = [samples[i] for i in va_idx]
test = [samples[i] for i in te_idx]
if args.model == "timemae":
    off, patch = samples[0]["inject_offset"], encoder_for(args.target).patch_size
    if off % patch:
        sys.exit(f"patch {patch} tidak membagi inject_offset {off}: patch melintasi batas pra/pasca-injeksi.")
print(f"target {args.target}: train/val/test = {len(train_full)}/{len(val)}/{len(test)} | sumber {sources}"
      f" | device {device}")


def subsample_train(train, fraction, seed):
    """Subset terstratifikasi per (sistem, root service), minimal 1 kasus per grup."""
    if fraction >= 1:
        return list(train)
    rng = np.random.default_rng(1000 + seed)
    groups = {}
    for i, s in enumerate(train):
        groups.setdefault((s["system"], s["root_service"]), []).append(i)
    keep = []
    for key in sorted(groups):
        idx = rng.permutation(groups[key])
        keep.extend(idx[:max(1, int(round(fraction * len(idx))))])
    return [train[i] for i in sorted(keep)]


def load_encoder(re_name, seed):
    enc = encoder_for(re_name)
    # map_location: checkpoint EX-1 disimpan dari MPS
    enc.load_state_dict(torch.load(args.ckpt_dir / f"{re_name}_{args.model}_seed{seed}.pt", map_location="cpu"))
    return enc.to(device).eval()


def make_encoders(seed):
    """(variant, source, encoder). Scratch diinisialisasi persis seperti run_ex1.py."""
    out = [("transfer", s, lambda s=s: load_encoder(s, seed)) for s in sources]
    if not args.no_indomain:
        out.append(("indomain", args.target, lambda: load_encoder(args.target, seed)))
    if not args.no_scratch:
        def scratch():
            torch.manual_seed(seed)
            return encoder_for(args.target).to(device).eval()
        out.append(("scratch", "-", scratch))
    return out


summary_path, pred_path, hist_path = out_dir / "summary.csv", out_dir / "predictions.csv", out_dir / "history.csv"
KEY = ["variant", "source", "mode", "seed", "fraction"]
summary_rows, pred_rows, hist_rows = [], [], []
done = set()
if args.resume and summary_path.exists():
    old = pd.read_csv(summary_path)
    summary_rows = old.to_dict("records")
    pred_rows = [pd.read_csv(pred_path)] if pred_path.exists() else []
    hist_rows = [pd.read_csv(hist_path)] if hist_path.exists() else []
    done = {tuple(r[k] for k in KEY) for r in summary_rows}
    print(f"resume: {len(done)} kombinasi sudah ada")


def save():
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    if pred_rows:
        pd.concat(pred_rows).to_csv(pred_path, index=False)
    if hist_rows:
        pd.concat(hist_rows).to_csv(hist_path, index=False)


TRAINERS = {
    "FineLast": lambda e, tr, sd: train_finelast(e, tr, val, device, seed=sd),
    "FineAll": lambda e, tr, sd: train_fineall(e, tr, val, device, epochs=args.fineall_epochs, seed=sd),
}

for frac in args.fractions:
    for seed in args.seeds:
        train = subsample_train(train_full, frac, seed)
        print(f"\n=== fraksi {frac:g} (n_train {len(train)}), seed {seed} ===")
        for variant, src, build in make_encoders(seed):
            todo = [m for m in args.modes if (variant, src, m, seed, frac) not in done]
            if not todo:
                continue
            enc = build()
            for mode in todo:
                t = time.time()
                r = TRAINERS[mode](enc, train, seed)
                secs = time.time() - t
                pt = predict_table(r["encoder"], r["head"], test, device)
                tags = dict(variant=variant, source=src, mode=mode, seed=seed, fraction=frac)
                stats = summarize_ranks(pt["rank"])
                summary_rows.append(tags | dict(n_train=len(train), train_sec=round(secs, 1),
                                                epochs=len(r["history"])) | stats)
                pred_rows.append(pt.assign(**tags))
                hist_rows.append(r["history"].assign(**tags))
                print(f"  {variant:8s} {src:3s} {mode:8s}: " +
                      " ".join(f"{k}={v:.3f}" for k, v in stats.items() if k != "n") + f" ({secs:.0f}s)", flush=True)
                save()                                             # simpan bertahap

heur = heuristic_table(test).assign(variant="heuristic", source="-", mode="-", seed=-1, fraction=np.nan)
heur.to_csv(out_dir / "heuristic_test.csv", index=False)

summary = pd.DataFrame(summary_rows)
agg = (summary.groupby(["fraction", "mode", "variant", "source"])[["AC@1", "AC@3", "Avg@5"]]
       .agg(["mean", "std"]).round(3))
print(f"\n== Ringkasan test {args.target} (mean ± std antar seed) ==")
print(agg.to_string())
print("\nHeuristik (test):", {k: round(v, 3) for k, v in summarize_ranks(heur["rank"]).items()})
