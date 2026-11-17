"""Unduh RCAEval (RE1/RE2/RE3) dari Hugging Face ke layout yang dipakai src/ssl_rcl/data.py.

Layout hasil:  dataset/RE{n}/<nama_kasus>/{metrics.parquet, inject_time.txt}

Default hanya mengunduh metrics.parquet dan inject_time.txt (cukup untuk pipeline,
±270 MB). Pakai --all-files untuk ikut mengunduh logs.parquet dan traces.parquet (±3.4 GB).

    python scripts/download_dataset.py                     # ke ./dataset
    python scripts/download_dataset.py --data-dir ~/datasets/rcaeval   # lalu buat symlink ./dataset
    python scripts/download_dataset.py --re RE1 --all-files

Kalau repo meminta login: `hf auth login` (atau `huggingface-cli login`), atau set env HF_TOKEN.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_rcl.data import list_cases  # noqa: E402

REPO_ID = "phamquiluan/RCAEval"
# Jumlah kasus valid (setelah EXCLUDED_CASES) di mesin lama.
EXPECTED = {"RE1": 373, "RE2": 270, "RE3": 90}

p = argparse.ArgumentParser()
p.add_argument("--re", nargs="+", default=["RE1", "RE2", "RE3"], choices=list(EXPECTED))
p.add_argument("--data-dir", type=Path, default=ROOT / "dataset",
               help="folder tujuan; kalau bukan ./dataset, symlink ./dataset dibuat otomatis")
p.add_argument("--all-files", action="store_true", help="ikut unduh logs.parquet dan traces.parquet")
p.add_argument("--workers", type=int, default=4)
args = p.parse_args()

data_dir = args.data_dir.expanduser().resolve()
data_dir.mkdir(parents=True, exist_ok=True)

for re_name in args.re:
    prefix = re_name.lower()  # di repo HF kasus ada di root: re1ob_adservice_cpu_1/...
    patterns = [f"{prefix}*"] if args.all_files else [f"{prefix}*/metrics.parquet", f"{prefix}*/inject_time.txt"]
    print(f"== {re_name}: {patterns} -> {data_dir / re_name}")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=data_dir / re_name,
                      allow_patterns=patterns, max_workers=args.workers,
                      token=os.environ.get("HF_TOKEN"))

link = ROOT / "dataset"
if data_dir != link.resolve():
    if link.is_symlink() or not link.exists():
        if link.is_symlink():
            link.unlink()
        link.symlink_to(data_dir, target_is_directory=True)
        print(f"symlink {link} -> {data_dir}")
    else:
        print(f"PERINGATAN: {link} sudah ada dan bukan symlink; tidak diubah.")

ok = True
for re_name in args.re:
    n = len(list_cases(data_dir, [re_name]))
    status = "OK" if n == EXPECTED[re_name] else f"BEDA (harusnya {EXPECTED[re_name]})"
    ok &= n == EXPECTED[re_name]
    print(f"{re_name}: {n} kasus valid  {status}")
sys.exit(0 if ok else 1)
