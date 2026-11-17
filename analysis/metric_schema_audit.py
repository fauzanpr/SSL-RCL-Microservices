"""Audit kolom metrik RCAEval RE1/RE2/RE3 untuk menentukan skema fitur bersama.

Jalankan dari root proyek:
    python analysis/metric_schema_audit.py

Output (CSV) ditulis ke report/metric_audit/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
OUT = ROOT / "report" / "metric_audit"
OUT.mkdir(parents=True, exist_ok=True)

CASE_RE = re.compile(r"^re(?P<re>\d)(?P<sys>ob|ss|tt)_(?P<svc>.+)_(?P<fault>[a-z0-9]+)_(?P<rep>\d+)$")


def split_col(col: str) -> tuple[str, str]:
    """'ts-auth-service_latency-90' -> ('ts-auth-service', 'latency-90')."""
    svc, _, metric = col.rpartition("_")
    return svc, metric


case_rows, col_rows = [], []
for re_dir in ["RE1", "RE2", "RE3"]:
    for case_dir in sorted((DATA / re_dir).iterdir()):
        m = CASE_RE.match(case_dir.name)
        if not m or not (case_dir / "metrics.parquet").exists():
            continue
        df = pd.read_parquet(case_dir / "metrics.parquet")
        t = df["time"].to_numpy()
        inject = int((case_dir / "inject_time.txt").read_text().strip())
        X = df.drop(columns="time")

        nan_frac = X.isna().mean()
        std = X.std(skipna=True)
        root_cols = [c for c in X.columns if split_col(c)[0] == m["svc"]]

        case_rows.append(dict(
            case=case_dir.name, re=f"RE{m['re']}", system=m["sys"], root_service=m["svc"],
            fault=m["fault"], rep=int(m["rep"]),
            n_rows=len(df), n_cols=X.shape[1],
            dt_median=float(np.median(np.diff(t))) if len(t) > 1 else np.nan,
            duration_s=int(t[-1] - t[0]),
            pre_inject_s=int(inject - t[0]), post_inject_s=int(t[-1] - inject),
            nan_frac=float(X.isna().to_numpy().mean()),
            const_col_frac=float(((std.fillna(0) == 0)).mean()),
            root_in_metrics=len(root_cols) > 0,
            root_metric_types=",".join(sorted(split_col(c)[1] for c in root_cols)),
        ))
        for c in X.columns:
            svc, metric = split_col(c)
            col_rows.append(dict(
                case=case_dir.name, re=f"RE{m['re']}", system=m["sys"], service=svc, metric=metric,
                nan_frac=float(nan_frac[c]), constant=bool(np.nan_to_num(std[c]) == 0),
            ))

cases = pd.DataFrame(case_rows)
cols = pd.DataFrame(col_rows)
cases.to_csv(OUT / "cases.csv", index=False)
cols.to_csv(OUT / "columns_long.csv", index=False)

n_cases = cases.groupby(["re", "system"]).size().rename("n_cases")

# 1) Cakupan jenis metrik: proporsi kasus yang punya metrik tipe ini (minimal di satu service)
mt = (cols.groupby(["re", "system", "metric"])["case"].nunique()
      .div(n_cases).unstack(["re", "system"]).fillna(0).round(2))
mt.to_csv(OUT / "metric_type_coverage.csv")

# 2) Cakupan (service, metric) per (re, system)
sm = (cols.groupby(["re", "system", "service", "metric"])["case"].nunique()
      .div(n_cases).rename("coverage").reset_index())
sm.to_csv(OUT / "service_metric_coverage.csv", index=False)

# 3) Service x jenis metrik: berapa jenis metrik per service (coverage >= 0.9)
stable = sm[sm.coverage >= 0.9]
svc_tbl = (stable.groupby(["re", "system", "service"])["metric"]
           .apply(lambda s: ",".join(sorted(s))).reset_index())
svc_tbl.to_csv(OUT / "service_metric_types.csv", index=False)

# 4) Irisan service antar RE per sistem (service yang punya cpu stabil)
core = stable[stable.metric == "cpu"]
overlap = []
for sys_ in ["ob", "ss", "tt"]:
    sets = {r: set(core[(core.system == sys_) & (core.re == r)].service) for r in ["RE1", "RE2", "RE3"]}
    overlap.append(dict(system=sys_, **{f"{r}_n": len(s) for r, s in sets.items()},
                        all_three=len(set.intersection(*sets.values())),
                        re2_re3=len(sets["RE2"] & sets["RE3"])))
pd.DataFrame(overlap).to_csv(OUT / "service_overlap.csv", index=False)

# 5) Statistik kualitas per kolom metrik
quality = (cols.groupby(["re", "metric"])
           .agg(n=("case", "size"), nan_frac=("nan_frac", "mean"), constant_frac=("constant", "mean"))
           .round(3))
quality.to_csv(OUT / "metric_quality.csv")

pd.set_option("display.width", 200, "display.max_rows", 200, "display.max_columns", 30)
print("== Ringkasan kasus ==")
print(cases.groupby(["re", "system"]).agg(
    n=("case", "size"), rows_med=("n_rows", "median"), dt=("dt_median", "median"),
    dur_min=("duration_s", "min"), dur_med=("duration_s", "median"), dur_max=("duration_s", "max"),
    pre_min=("pre_inject_s", "min"), pre_med=("pre_inject_s", "median"),
    post_min=("post_inject_s", "min"), post_med=("post_inject_s", "median"),
    nan=("nan_frac", "mean"), const=("const_col_frac", "mean"), root_ok=("root_in_metrics", "mean"),
).round(3))
print("\n== Cakupan jenis metrik (proporsi kasus) ==")
print(mt)
print("\n== Kualitas per jenis metrik ==")
print(quality)
print("\n== Overlap service (yang punya cpu stabil) ==")
print(pd.DataFrame(overlap))
print("\n== Kasus dengan root service tidak ada di metrik ==")
print(cases.loc[~cases.root_in_metrics, ["case"]].to_string(index=False) or "(tidak ada)")
print("\n== Jenis metrik yang tersedia untuk root service ==")
print(cases.groupby(["re", "system"])["root_metric_types"].agg(lambda s: s.value_counts().head(3).to_dict()))
