"""Cek progress EX-1 / EX-2 yang dijalankan lewat scripts/launch_ex1.py.

    python scripts/check_ex1.py                    # ringkasan sekali jalan
    python scripts/check_ex1.py --watch 30         # refresh tiap 30 detik (Ctrl+C untuk keluar)
    python scripts/check_ex1.py --tail 15          # plus 15 baris terakhir log RE yang sedang jalan
    python scripts/check_ex1.py --model ts2vec
    python scripts/check_ex1.py --exp ex2          # pantau EX-2
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd

from _ex1_common import ROOT, log_path, pid_alive, read_status

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

p = argparse.ArgumentParser()
p.add_argument("--exp", default="ex1", choices=["ex1", "ex2"])
p.add_argument("--model", default="timemae", choices=["ts2vec", "timemae", "run"])
p.add_argument("--watch", type=int, default=0, help="refresh tiap N detik")
p.add_argument("--tail", type=int, default=0, help="tampilkan N baris terakhir log RE yang aktif")
args = p.parse_args()

STEPS_PER_SEED = 5                                       # pre-training + (pretrain, scratch) x (FineLast, FineAll)
RES_RE = re.compile(r"^\s+(pretrain|scratch)\s+(FineLast|FineAll)\s+seed (\d+): (.*)\((\d+)s\)")
RES2_RE = re.compile(r"^\s+(transfer|indomain|scratch)\s+(\S+)\s+(FineLast|FineAll)\s*: (.*)\((\d+)s\)")
ITER_RE = re.compile(r"iter\s+(\d+)/(\d+)\s+(.*)")


def fmt_td(sec):
    sec = int(sec)
    return f"{sec // 3600}j {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def parse_log(path, total):
    """-> dict(frac, seed, stage, results, error)."""
    info = dict(frac=0.0, seed=None, stage="menunggu", results=[], error=None, pt=None)
    if not path.exists():
        return info
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    done_steps, seed, stage, pt_frac = 0, None, "memuat data", 0.0
    for ln in lines:
        if ln.startswith("=== fraksi"):                 # EX-2
            stage = ln.strip("= ").strip(); seed = re.search(r"seed (\d+)", ln)[1]
        elif (m := RES2_RE.match(ln)):
            done_steps += 1; info["results"].append((m[1], m[3], m[2], m[4].strip(), int(m[5])))
        elif ln.startswith("=== seed"):
            seed = int(ln.split()[2]); stage = "pre-training"; pt_frac = 0.0
        elif (m := ITER_RE.search(ln)):
            pt_frac = int(m[1]) / int(m[2]); info["pt"] = f"iter {m[1]}/{m[2]}  {m[3].strip()}"
        elif "pre-training selesai" in ln:
            done_steps += 1; pt_frac = 0.0; stage = "fine-tuning"
        elif (m := RES_RE.match(ln)):
            done_steps += 1; info["results"].append((m[1], m[2], int(m[3]), m[4].strip(), int(m[5])))
            stage = f"fine-tuning ({m[1]} {m[2]} selesai)"
        elif ln.startswith("Traceback") or re.match(r"^\w*(Error|Exception)", ln):
            info["error"] = True
    if info["error"]:
        info["error"] = "\n".join(lines[-12:])
    info["seed"], info["stage"] = seed, stage
    info["frac"] = min((done_steps + pt_frac) / total, 1.0)
    if any("Ringkasan test" in ln for ln in lines):
        info["frac"], info["stage"] = 1.0, "selesai"
    return info


def gpu_line():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        n, u, mu, mt = [x.strip() for x in out.split(",")]
        return f"{n} | util {u}% | memori {mu}/{mt} MiB"
    except Exception:
        return "nvidia-smi tidak tersedia"


def result_table(src, model):
    d = ROOT / "results" / args.exp / f"{src}_{model}"
    f = d / "summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    keys = ["variant", "mode"] if args.exp == "ex1" else ["fraction", "mode", "variant", "source"]
    agg = s.groupby(keys)["AC@1"].agg(["mean", "std", "count"])
    rows = []
    for k, r in agg.iterrows():
        label = f"{k[0]:9s} {k[1]:8s}" if args.exp == "ex1" else f"frak {k[0]:<4g} {k[1]:8s} {k[2]:8s} {k[3]:3s}"
        rows.append(f"    {label} AC@1 {r['mean']:.3f} ± {0 if pd.isna(r['std']) else r['std']:.3f}  (seed {int(r['count'])})")
    h = d / "heuristic_test.csv"
    if h.exists():
        hd = pd.read_csv(h)
        rows.append(f"    heuristik            AC@1 {(hd['rank'] <= 1).mean():.3f}  (n {len(hd)})")
    return "\n".join(rows)


def show():
    st = read_status(args.model, args.exp)
    print(f"=== {args.exp.upper()} {args.model} | {datetime.now():%Y-%m-%d %H:%M:%S} ===")
    if not st:
        print("Belum pernah dijalankan. Mulai dengan: python scripts/launch_ex1.py --exp " + args.exp); return False
    state = st["state"]
    alive = pid_alive(st.get("worker_pid"))
    if state in ("running", "starting") and not alive:
        state = "BERHENTI TIBA-TIBA (proses tidak ditemukan; cek log)"
    t0 = datetime.fromisoformat(st["started"])
    end = datetime.fromisoformat(st["finished"]) if st.get("finished") else datetime.now()
    print(f"Status : {state} | mulai {st['started']} | berjalan {fmt_td((end - t0).total_seconds())}")
    print(f"Rencana: {' -> '.join(st['sources'])} | seed {st['seeds']}" + (f" | opsi {st['extra']}" if st["extra"] else ""))
    print(f"GPU    : {gpu_line()}\n")
    active = None
    for src in st["sources"]:
        run = st["runs"].get(src)
        if not run:
            print(f"[{src}] menunggu"); continue
        total = (st.get("total") or {}).get(src) or STEPS_PER_SEED * len(st["seeds"])
        info = parse_log(log_path(src, args.model, args.exp), total)
        r_end = datetime.fromisoformat(run["finished"]) if run.get("finished") else datetime.now()
        el = (r_end - datetime.fromisoformat(run["started"])).total_seconds()
        bar = "#" * int(info["frac"] * 20) + "-" * (20 - int(info["frac"] * 20))
        if run["state"] == "running" and not alive:
            run = dict(run, state="stopped")            # worker mati tapi status belum sempat diperbarui
        tag = {"running": "JALAN", "done": "SELESAI", "failed": f"GAGAL (exit {run['exit_code']})",
               "stopped": "TERHENTI/DIHENTIKAN"}[run["state"]]
        eta = ""
        if run["state"] == "running" and info["frac"] > 0.03:
            eta = f" | sisa ±{fmt_td(el * (1 - info['frac']) / info['frac'])} (kasar)"
        print(f"[{src}] {tag} [{bar}] {info['frac']:.0%} | {fmt_td(el)}{eta}")
        if run["state"] == "running":
            active = src
            print(f"    seed {info['seed']} | {info['stage']}")
            if info["pt"] and "pre-training" in info["stage"]:
                print(f"    {info['pt']}")
        if info["error"] and run["state"] in ("failed", "running"):
            print("    ! terdeteksi error di log:\n      " + info["error"].replace("\n", "\n      "))
        tbl = result_table(src, args.model)
        if tbl:
            print("    hasil sementara (test)" if run["state"] == "running" else "    hasil (test)")
            print(tbl)
    if args.tail and active:
        print(f"\n--- {args.tail} baris terakhir logs/{args.exp}_{active}_{args.model}.log ---")
        lines = log_path(active, args.model, args.exp).read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-args.tail:]))
    return state in ("running", "starting")


if args.watch:
    try:
        while True:
            print("\033[2J\033[H", end="") if sys.stdout.isatty() else print("\n" + "=" * 70)
            if not show():
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
else:
    show()
