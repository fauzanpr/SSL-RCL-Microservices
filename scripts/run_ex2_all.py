"""Jalankan EX-2 (transfer lintas RE) untuk semua metode SSL dan semua target dalam satu perintah.

    python scripts/run_ex2_all.py                         # ts2vec, timemae, run x target RE1, RE2 (pengaturan penuh)
    python scripts/run_ex2_all.py --models run --targets RE2 --seeds 0       # sebagian saja
    python scripts/run_ex2_all.py --resume                # lanjutkan run yang terhenti (tanpa menimpa hasil)
    python scripts/run_ex2_all.py --dry-run               # hanya cetak perintah

Pengaturan default = pengaturan penuh EX-2: fraksi label 0.1 0.25 0.5 1.0, mode FineLast + FineAll, seed 0 1 2.
Varian per target: transfer (tiap RE sumber lain), indomain, scratch. RE3 tidak jadi target default karena sudah
ceiling (AC@1 ~ 1.0 bahkan untuk scratch); tambahkan lewat --targets RE1 RE2 RE3 bila perlu.

Sebelum jalan, skrip mengecek checkpoint EX-1 (checkpoints/ex1/<RE>_<model>_seed<s>.pt) untuk SEMUA RE. Yang hilang
dibuat ulang lewat `run_ex1.py --pretrain-only` (hanya pre-training, results/ex1 tidak disentuh), kecuali --no-pretrain.

Tanpa --resume, results/ex2/<target>_<model>/ ditimpa (hasil lama ada di git).
Log per run : logs/ex2_<target>_<model>.log     Hasil: results/ex2/<target>_<model>/
Satu run yang gagal tidak menghentikan run lain; kode keluar != 0 bila ada yang gagal.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALL_RE = ["RE1", "RE2", "RE3"]
MODELS = ["ts2vec", "timemae", "run"]

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
p.add_argument("--targets", nargs="+", default=["RE1", "RE2"], choices=ALL_RE)
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
p.add_argument("--modes", nargs="+", default=["FineLast", "FineAll"], choices=["FineLast", "FineAll"])
p.add_argument("--resume", action="store_true", help="lewati kombinasi yang sudah ada di summary.csv")
p.add_argument("--no-pretrain", action="store_true", help="jangan buat ulang checkpoint yang hilang (gagal bila ada)")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

now = lambda: datetime.now().isoformat(timespec="seconds")  # noqa: E731
(ROOT / "logs").mkdir(exist_ok=True)


def run(cmd, log_path):
    print(f"[{now()}] {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return 0
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# {now()}  {' '.join(cmd)}\n"); lf.flush()
        env = dict(__import__("os").environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        return subprocess.call(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)


def ckpt(re_name, model, seed):
    return ROOT / "checkpoints" / "ex1" / f"{re_name}_{model}_seed{seed}.pt"


failed, t_all = [], time.time()
for model in args.models:
    # 1. checkpoint EX-1 untuk semua RE (sumber transfer + indomain)
    for re_name in ALL_RE:
        miss = [s for s in args.seeds if not ckpt(re_name, model, s).exists()]
        if not miss:
            continue
        if args.no_pretrain:
            sys.exit(f"Checkpoint hilang: {re_name}_{model} seed {miss}")
        code = run([sys.executable, "-u", str(ROOT / "scripts" / "run_ex1.py"), "--source", re_name, "--model", model,
                    "--seeds", *map(str, miss), "--pretrain-only"], ROOT / "logs" / f"ex2_prep_{re_name}_{model}.log")
        if code:
            failed.append(f"pretrain {re_name} {model}")

    # 2. EX-2 per target
    for target in args.targets:
        t = time.time()
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "run_ex2.py"), "--target", target, "--model", model,
               "--seeds", *map(str, args.seeds), "--fractions", *map(str, args.fractions),
               "--modes", *args.modes] + (["--resume"] if args.resume else [])
        code = run(cmd, ROOT / "logs" / f"ex2_{target}_{model}.log")
        print(f"[{now()}] {model} -> {target}: {'selesai' if code == 0 else f'GAGAL (kode {code})'} "
              f"({(time.time() - t) / 60:.1f} menit)", flush=True)
        if code:
            failed.append(f"ex2 {target} {model}")

print(f"\n[{now()}] selesai dalam {(time.time() - t_all) / 60:.1f} menit. Gagal: {failed or 'tidak ada'}")
sys.exit(1 if failed else 0)
