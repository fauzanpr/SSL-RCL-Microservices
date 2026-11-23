"""Uji berpasangan AC@1: setiap varian encoder dibandingkan dengan scratch pada kasus test yang sama.

    python scripts/paired_test.py --target RE2                 # results/ex2/RE2_ts2vec/
    python scripts/paired_test.py --target RE1 --model ts2vec --n-boot 10000

Metode: bootstrap berpasangan per kasus.
  1. Untuk setiap kasus test, hit = 1 jika rank root-cause = 1. Hit dirata-rata atas seed, sehingga
     setiap kasus bernilai 0, 1/3, 2/3, atau 1. Dengan begitu kasus yang sama tidak dihitung 3 kali.
  2. Selisih per kasus = hit(varian) - hit(scratch). Rata-ratanya sama dengan selisih AC@1 mean antar seed.
  3. Kasus test di-resample dengan pengembalian sebanyak --n-boot kali, menghasilkan CI 95% (persentil)
     untuk selisih tersebut. Jika CI tidak mencakup 0, selisihnya dianggap bermakna.
Sebagai pelengkap deskriptif: jumlah pasangan (kasus, seed) yang hanya benar di varian (`only_variant`)
atau hanya benar di scratch (`only_scratch`).

Hasil: results/ex2/<target>_<model>/paired_test.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--target", required=True)
p.add_argument("--model", default="ts2vec")
p.add_argument("--baseline", default="scratch", help="varian pembanding")
p.add_argument("--n-boot", type=int, default=10000)
p.add_argument("--seed", type=int, default=0, help="seed RNG bootstrap")
p.add_argument("--res-dir", type=Path, default=None, help="default: results/ex2/<target>_<model>")
args = p.parse_args()

res_dir = args.res_dir or ROOT / "results" / "ex2" / f"{args.target}_{args.model}"
pred = pd.read_csv(res_dir / "predictions.csv")
pred["hit"] = (pred["rank"] == 1).astype(float)
pred["arm"] = pred["variant"] + ":" + pred["source"].astype(str)
base_arm = f"{args.baseline}:-"
rng = np.random.default_rng(args.seed)

rows = []
for (mode, frac), g in pred.groupby(["mode", "fraction"]):
    # matriks hit [kasus x seed] per arm; kasus diurutkan berdasarkan nama supaya berpasangan
    wide = g.pivot_table(index=["name", "seed"], columns="arm", values="hit")
    if base_arm not in wide:
        continue
    for arm in wide.columns:
        if arm == base_arm:
            continue
        pair = wide[[arm, base_arm]].dropna()
        per_case = pair.groupby(level="name").mean()                  # rata-rata atas seed
        d = (per_case[arm] - per_case[base_arm]).to_numpy()
        idx = rng.integers(0, len(d), size=(args.n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        variant, source = arm.split(":")
        rows.append(dict(
            mode=mode, fraction=frac, variant=variant, source=source,
            n_cases=len(d), n_seeds=pair.index.get_level_values("seed").nunique(),
            AC1_variant=per_case[arm].mean(), AC1_scratch=per_case[base_arm].mean(),
            diff=d.mean(), ci_low=lo, ci_high=hi,
            significant=bool(lo > 0 or hi < 0),
            only_variant=int(((pair[arm] == 1) & (pair[base_arm] == 0)).sum()),
            only_scratch=int(((pair[arm] == 0) & (pair[base_arm] == 1)).sum()),
        ))

out = pd.DataFrame(rows).sort_values(["mode", "fraction", "variant", "source"])
out.to_csv(res_dir / "paired_test.csv", index=False)

show = out.copy()
for c in ["AC1_variant", "AC1_scratch", "diff", "ci_low", "ci_high"]:
    show[c] = show[c].round(3)
show["CI95"] = "[" + show["ci_low"].map("{:+.3f}".format) + ", " + show["ci_high"].map("{:+.3f}".format) + "]"
print(f"== Uji berpasangan {args.target} ({args.model}), pembanding: {args.baseline}, "
      f"bootstrap {args.n_boot}x ==")
print(show[["mode", "fraction", "variant", "source", "AC1_variant", "AC1_scratch", "diff", "CI95",
            "significant", "only_variant", "only_scratch"]].to_string(index=False))
print(f"\nDisimpan: {res_dir / 'paired_test.csv'}")
