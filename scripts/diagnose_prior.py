"""Diagnostik: apakah model mendeteksi ANOMALI, atau hanya mengenali service yang SERING jadi root?

Dua uji di split test (encoder pretrain seed 0 + FineLast yang dilatih ulang):
  1. Prior frekuensi: ranking service menurut seberapa sering ia menjadi root di split train
     (tanpa melihat data metrik kasus test sama sekali).
  2. Anomali dihapus: segmen pasca-injeksi diganti dengan potongan data PRA-injeksi dari kasus
     yang sama, sehingga tidak ada gangguan. Model yang benar-benar mendeteksi anomali seharusnya
     turun ke level acak; jika AC@1 tetap tinggi, model memakai identitas service.

    python scripts/diagnose_prior.py --sources RE1 RE2 RE3
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssl_rcl.data import CORE_METRICS, build_samples, stratified_split  # noqa: E402
from ssl_rcl.finetune import evaluate_cases, summarize_ranks, train_finelast  # noqa: E402
from ssl_rcl.models.ts2vec import TSEncoder  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--sources", nargs="+", default=["RE1", "RE2", "RE3"])
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
K = len(CORE_METRICS)


def remove_anomaly(s):
    """Ganti pasca-injeksi dengan data pra-injeksi (tanpa gangguan) dari kasus yang sama."""
    off, T = s["inject_offset"], s["x"].shape[1]
    post_len = T - off
    x = s["x"].copy()
    x[:, off:] = s["x"][:, off - post_len:off]
    return dict(s, x=x)


rows = []
for src in args.sources:
    samples = build_samples(ROOT / "dataset", [src], cache_dir=ROOT / "cache", verbose=False)
    tr, va, te = stratified_split(samples, seed=42)
    train, val, test = [samples[i] for i in tr], [samples[i] for i in va], [samples[i] for i in te]

    # 1) prior frekuensi root dari train (per sistem). Skor seri (mis. 5 root dengan frekuensi
    #    sama) diurai secara acak -> dihitung peluang harapan P(rank <= k), bukan rank optimis.
    freq = {sys_: Counter(s["root_service"] for s in train if s["system"] == sys_) for sys_ in {"ob", "ss", "tt"}}
    prior_hit = {k: [] for k in range(1, 6)}
    for s in test:
        sc = np.array([freq[s["system"]].get(n, 0) for n in s["services"]], float)
        g = int((sc > sc[s["label"]]).sum())                  # service yang pasti di atas
        t = int((sc == sc[s["label"]]).sum()) - 1             # service lain yang seri
        for k in prior_hit:
            prior_hit[k].append(np.clip((k - g) / (t + 1), 0, 1))

    # 2) model: encoder pretrain + FineLast
    enc = TSEncoder(2 * K)
    enc.load_state_dict(torch.load(ROOT / "checkpoints" / "ex1" / f"{src}_ts2vec_seed{args.seed}.pt"))
    enc = enc.to(device).eval()
    res = train_finelast(enc, train, val, device, seed=args.seed)
    _, r_normal = evaluate_cases(res["encoder"], res["head"], test, device)
    _, r_noanom = evaluate_cases(res["encoder"], res["head"], [remove_anomaly(s) for s in test], device)

    n_svc = np.mean([len(s["services"]) for s in test])
    n_root = {k: len(v) for k, v in freq.items()}
    pk = {f"AC@{k}": float(np.mean(prior_hit[k])) for k in (1, 3, 5)}
    pk["Avg@5"] = float(np.mean([np.mean(prior_hit[k]) for k in range(1, 6)]))
    rows.append(dict(source=src, uji="prior frekuensi (tanpa data)", **pk, n=len(test)))
    rows.append(dict(source=src, uji="acak (1/N)", **{f"AC@{k}": float(np.mean([min(k / len(s["services"]), 1) for s in test]))
                                                      for k in (1, 3, 5)}, n=len(test)))
    for name, r in [("model, data normal", r_normal), ("model, anomali dihapus", r_noanom)]:
        rows.append(dict(source=src, uji=name, **summarize_ranks(r)))
    print(f"{src}: rata-rata service/kasus {n_svc:.1f}, jumlah root berbeda di train per sistem {n_root}")

out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv(ROOT / "results" / "ex1" / "diagnose_prior.csv", index=False)
