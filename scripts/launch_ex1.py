"""Jalankan EX-1 atau EX-2 di background (tetap berjalan walau terminal/VS Code ditutup), RE demi RE secara berurutan.

    python scripts/launch_ex1.py                                  # timemae, RE1 -> RE2 -> RE3, seed 0 1 2
    python scripts/launch_ex1.py --sources RE1 --seeds 0          # sebagian saja
    python scripts/launch_ex1.py --model ts2vec
    python scripts/launch_ex1.py -- --n-iters 5000 --tm-gamma 2   # argumen setelah "--" diteruskan ke run_ex1.py
    python scripts/launch_ex1.py --stop                           # hentikan proses yang sedang jalan

    # EX-2 (transfer; butuh checkpoint EX-1). --sources = daftar TARGET, default RE1 RE2 (RE3 sudah ceiling)
    python scripts/launch_ex1.py --exp ex2
    python scripts/launch_ex1.py --exp ex2 -- --fractions 0.1 0.25 0.5 1.0 --modes FineLast
    python scripts/launch_ex1.py --exp ex2 --stop

Log per RE : logs/<exp>_<RE>_<model>.log
Status     : logs/<exp>_<model>_status.json   (dibaca oleh scripts/check_ex1.py)
Hasil      : results/<exp>/<RE>_<model>/    (ditulis bertahap)
PERINGATAN : menjalankan ulang RE yang sama menimpa results/<exp>/<RE>_<model>/ (ex1: juga checkpoint-nya).
            EX-2 tanpa --resume juga menimpa; tambahkan "-- --resume" untuk melanjutkan.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from _ex1_common import (IS_WIN, LOG_DIR, ROOT, kill_tree, log_path, pid_alive, read_status, status_path,
                         write_status)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", default="timemae", choices=["ts2vec", "timemae", "run"])
p.add_argument("--exp", default="ex1", choices=["ex1", "ex2"])
p.add_argument("--sources", nargs="+", default=None, help="ex1: RE sumber; ex2: RE target (default ex1 RE1 RE2 RE3, ex2 RE1 RE2)")
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--stop", action="store_true", help="hentikan proses background yang sedang berjalan")
p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)   # dipakai internal
p.add_argument("extra", nargs=argparse.REMAINDER, help="argumen tambahan untuk run_ex1.py (setelah --)")
args = p.parse_args()
extra = [a for a in args.extra if a != "--"]
args.sources = args.sources or (["RE1", "RE2", "RE3"] if args.exp == "ex1" else ["RE1", "RE2"])
EXP = args.exp
now = lambda: datetime.now().isoformat(timespec="seconds")  # noqa: E731


def worker():
    """Jalan di proses terpisah: panggil run_ex1.py untuk tiap RE, tulis status."""
    st = read_status(args.model, EXP)
    st.update(state="running", worker_pid=os.getpid())
    write_status(args.model, st, EXP)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for src in args.sources:
        st["current"] = src
        st["runs"][src] = dict(state="running", started=now(), child_pid=None, exit_code=None)
        write_status(args.model, st, EXP)
        script, flag = ("run_ex1.py", "--source") if EXP == "ex1" else ("run_ex2.py", "--target")
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / script), flag, src, "--model", args.model,
               "--seeds", *map(str, args.seeds), *extra]
        with open(log_path(src, args.model, EXP), "w", encoding="utf-8") as lf:
            lf.write(f"# {now()}  {' '.join(cmd)}\n"); lf.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
            st["runs"][src]["child_pid"] = proc.pid
            write_status(args.model, st, EXP)
            code = proc.wait()
        st["runs"][src].update(state="done" if code == 0 else "failed", finished=now(), exit_code=code)
        write_status(args.model, st, EXP)                    # RE berikutnya tetap jalan walau satu RE gagal
    st.update(state="finished", current=None, finished=now())
    write_status(args.model, st, EXP)


def stop():
    st = read_status(args.model, EXP)
    if not st or st.get("state") != "running":
        print("Tidak ada proses yang sedang berjalan."); return
    kill_tree(st.get("worker_pid"))                     # worker dulu, supaya tidak menjalankan RE berikutnya
    for r in st["runs"].values():
        if r.get("state") == "running":
            kill_tree(r.get("child_pid")); r.update(state="stopped", finished=now())
    st.update(state="stopped", current=None, finished=now())
    write_status(args.model, st, EXP)
    print("Dihentikan. Hasil parsial yang sudah selesai tetap ada di results/.")


def launch():
    st = read_status(args.model, EXP)
    if st and st.get("state") == "running" and pid_alive(st.get("worker_pid")):
        sys.exit(f"Sudah ada proses {args.model} berjalan (PID {st['worker_pid']}). Pantau: python scripts/check_ex1.py "
                 f"--exp {EXP} --model {args.model}\nHentikan dulu dengan --stop kalau ingin mengulang.")
    for src in args.sources:                            # gagal cepat bila data belum ada
        if not (ROOT / "dataset" / src).is_dir():
            sys.exit(f"dataset/{src} tidak ditemukan.")
    LOG_DIR.mkdir(exist_ok=True)
    total = None
    if EXP == "ex2":                                    # jumlah run per target, untuk progress bar
        q = argparse.ArgumentParser(); q.add_argument("--fractions", nargs="+", type=float, default=[1.0])
        q.add_argument("--modes", nargs="+", default=["FineLast", "FineAll"]); q.add_argument("--sources", nargs="+")
        q.add_argument("--no-indomain", action="store_true"); q.add_argument("--no-scratch", action="store_true")
        q.add_argument("--resume", action="store_true")
        qa, _ = q.parse_known_args(extra)
        total = {t: len(qa.fractions) * len(args.seeds) * len(qa.modes) *
                    (len(qa.sources or [r for r in ("RE1", "RE2", "RE3") if r != t])
                     + (not qa.no_indomain) + (not qa.no_scratch)) for t in args.sources}
    write_status(args.model, dict(state="starting", exp=EXP, model=args.model, sources=args.sources, seeds=args.seeds,
                                  extra=extra, total=total, started=now(), worker_pid=None, current=None, runs={}), EXP)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--exp", EXP, "--model", args.model,
           "--sources", *args.sources, "--seeds", *map(str, args.seeds), "--", *extra]
    kw = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if IS_WIN:   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000
    else:
        kw["start_new_session"] = True
    subprocess.Popen(cmd, **kw)
    for _ in range(50):                                 # tunggu worker menulis PID-nya
        time.sleep(0.2)
        st = read_status(args.model, EXP)
        if st and st.get("worker_pid"):
            break
    else:
        sys.exit(f"Worker tidak mau start. Jalankan manual: python scripts/run_{EXP}.py --model " + args.model)
    print(f"Berjalan di background (PID {st['worker_pid']}): {args.model} | {' -> '.join(args.sources)} | seed {args.seeds}")
    print(f"Pantau progress : python scripts/check_ex1.py --exp {EXP} --model {args.model}   (tambah --watch 30 untuk auto-refresh)")
    print(f"Hentikan        : python scripts/launch_ex1.py --exp {EXP} --model {args.model} --stop")


if args.worker:
    worker()
elif args.stop:
    stop()
else:
    launch()
