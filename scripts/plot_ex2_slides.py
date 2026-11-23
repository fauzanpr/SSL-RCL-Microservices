"""Grafik slide EX-2 TS2Vec, dibuat langsung dari file hasil supaya angkanya pasti sesuai.

    python scripts/plot_ex2_slides.py

Output (PNG, 200 dpi) di report/figures/:
  ex2_hasil_re2.png    AC@1 RE2 per jumlah data latih (FineLast), slide "Hasil utama EX-2"
  ex2_uji_bootstrap.png selisih AC@1 vs encoder acak + rentang 95%, slide "Apakah bukan kebetulan?"
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "ex2"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palet: slot kategorikal 1-3 dari palet referensi dataviz (lolos uji CVD all-pairs) + abu netral
# untuk pembanding "acak". Teks memakai tinta netral, bukan warna seri.
SURFACE, INK, INK_2, INK_3, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8f8e88", "#e6e5e0"
NAVY = "#0d2b5b"                                    # warna judul di template slide
SERIES = {
    "transfer:RE1": ("Transfer dari RE1", "#2a78d6"),
    "transfer:RE3": ("Transfer dari RE3", "#eb6834"),
    "indomain:RE2": ("In-domain (RE2)", "#1baf7a"),
    "scratch:-": ("Acak (tanpa pre-training)", "#c9c8c2"),
}
LABELED = set(SERIES)                               # semua batang diberi angka (permintaan untuk slide)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": INK_3, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def fmt(v, sign=False):
    """Angka dengan koma desimal (gaya Indonesia)."""
    if abs(v) < 0.005:
        v = 0.0                                     # hindari "-0,00"
    s = f"{v:+.2f}" if sign and v != 0 else f"{v:.2f}"
    return s.replace(".", ",").replace("-", "−")


# ------------------------------------------------------------------ grafik 1: hasil utama RE2
s = pd.read_csv(RES / "RE2_ts2vec" / "summary.csv")
s = s[s["mode"] == "FineLast"]
s["arm"] = s["variant"] + ":" + s["source"].astype(str)
mean = s.groupby(["fraction", "arm"])["AC@1"].mean().unstack()
fracs = [0.25, 0.5, 1.0]
heur = pd.read_csv(RES / "RE2_ts2vec" / "heuristic_test.csv")
heur_ac1 = float((heur["rank"] == 1).mean())

fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
n, width, gap = len(SERIES), 0.19, 0.012
x = np.arange(len(fracs))
for i, (arm, (label, color)) in enumerate(SERIES.items()):
    xs = x + (i - (n - 1) / 2) * width
    vals = [mean.loc[f, arm] for f in fracs]
    ax.bar(xs, vals, width - gap, color=color, label=label, zorder=3,
           edgecolor=INK_3 if arm == "scratch:-" else color, linewidth=0.8)
    if arm in LABELED:
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.012, fmt(v), ha="center", va="bottom", fontsize=11.5,
                    color=INK, fontweight="bold" if arm == "transfer:RE1" else "normal")

ax.hlines(heur_ac1, -0.5, len(fracs) - 0.5, color=INK_2, linestyle=(0, (5, 4)), linewidth=1.3, zorder=4)
ax.set_xlim(-0.5, len(fracs) - 0.5 + 0.45)               # ruang di kanan untuk label heuristik
ax.text(len(fracs) - 0.5 + 0.03, heur_ac1, f"Heuristik\n{fmt(heur_ac1)}", ha="left", va="center",
        fontsize=11.5, color=INK_2)
ax.set_xticks(x, [f"{int(f * 100)}%" for f in fracs], fontsize=13)
ax.set_xlabel("Jumlah data latih berlabel", fontsize=13, labelpad=8)
ax.set_ylabel("AC@1 (data uji RE2)", fontsize=13)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2), [fmt(v) for v in np.arange(0, 1.01, 0.2)])
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("AC@1 di RE2 per jumlah data latih (FineLast, rata-rata 3 seed)", loc="left",
             fontsize=15, color=NAVY, fontweight="bold", pad=34)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=11.5,
          handlelength=1.2, columnspacing=1.4, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(OUT / "ex2_hasil_re2.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------ grafik 2: uji bootstrap
def rows(target, source):
    p = pd.read_csv(RES / f"{target}_ts2vec" / "paired_test.csv")
    p = p[(p["variant"] == "transfer") & (p["source"] == source)]
    order = {"FineLast": 0, "FineAll": 1}
    p = p.assign(o=p["mode"].map(order)).sort_values(["o", "fraction"])
    return [dict(label=f"{r['mode']} {int(r['fraction'] * 100)}%", diff=r["diff"], lo=r["ci_low"],
                 hi=r["ci_high"], sig=bool(r["significant"])) for _, r in p.iterrows()]


groups = [("Tujuan RE2, encoder dari RE1", rows("RE2", "RE1")),
          ("Tujuan RE1, encoder dari RE2", rows("RE1", "RE2"))]

SIG, NOSIG = "#2a78d6", INK_3
fig, ax = plt.subplots(figsize=(9, 7.2), dpi=200)
y, yticks, ylabels, headers = 0, [], [], []
for title, rs in groups:
    headers.append((y, title))
    y -= 1
    for r in rs:
        c = SIG if r["sig"] else NOSIG
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, linewidth=2.2, solid_capstyle="round", zorder=3)
        ax.plot(r["diff"], y, "o", markersize=9, zorder=4, markeredgewidth=2, markeredgecolor=c,
                markerfacecolor=c if r["sig"] else SURFACE)
        ax.text(0.205, y, f"{fmt(r['diff'], sign=True)}   {'bermakna' if r['sig'] else 'tidak'}",
                va="center", ha="left", fontsize=11.5, color=INK if r["sig"] else INK_2,
                fontweight="bold" if r["sig"] else "normal")
        yticks.append(y); ylabels.append(r["label"])
        y -= 1
    y -= 0.6

for hy, title in headers:
    ax.text(-0.15, hy, title, va="center", ha="left", fontsize=12.5, color=NAVY, fontweight="bold")
ax.axvline(0, color=INK_2, linewidth=1.2, zorder=2)
ax.set_yticks(yticks, ylabels, fontsize=12)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.15, 0.20)
ax.set_ylim(y + 0.6, 0.6)
xt = np.arange(-0.10, 0.201, 0.05)
ax.set_xticks(xt, [fmt(v, sign=True) if abs(v) > 1e-9 else "0" for v in xt])
ax.set_xlabel("Selisih AC@1 terhadap encoder acak (titik = selisih, garis = rentang 95%)",
              fontsize=11.5, labelpad=8)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.text(0.205, 0.55, "Selisih", ha="left", va="bottom", fontsize=11, color=INK_2)
ax.set_title("Garis tidak menyentuh 0 = bukan kebetulan", loc="left", fontsize=15, color=NAVY,
             fontweight="bold", pad=14)
fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.11)
fig.savefig(OUT / "ex2_uji_bootstrap.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------ grafik 3: EX-1 FineLast vs FineAll
EX1 = ROOT / "results" / "ex1"
datasets = ["RE1", "RE2", "RE3"]
ex1 = pd.concat([pd.read_csv(EX1 / f"{d}_ts2vec" / "summary.csv").assign(dataset=d) for d in datasets])
stat = ex1.groupby(["mode", "dataset", "variant"])["AC@1"].agg(["mean", "std"])
heur1 = {d: float((pd.read_csv(EX1 / f"{d}_ts2vec" / "heuristic_test.csv")["rank"] == 1).mean())
         for d in datasets}
n_test = ex1.groupby("dataset")["n"].first().to_dict()
EX1_SERIES = [("pretrain", "Pre-training TS2Vec", "#2a78d6"), ("scratch", "Acak (tanpa pre-training)", "#c9c8c2")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=200, sharey=True)
width, gap = 0.34, 0.012
x = np.arange(len(datasets))
for ax, mode, sub in zip(axes, ["FineLast", "FineAll"],
                         ["encoder dibekukan, hanya head dilatih", "encoder ikut dilatih"]):
    for i, (var, label, color) in enumerate(EX1_SERIES):
        xs = x + (i - 0.5) * width
        m = [stat.loc[(mode, d, var), "mean"] for d in datasets]
        sd = [stat.loc[(mode, d, var), "std"] for d in datasets]
        ax.bar(xs, m, width - gap, color=color, label=label, zorder=3,
               edgecolor=INK_3 if var == "scratch" else color, linewidth=0.8)
        ax.errorbar(xs, m, yerr=sd, fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4, zorder=4)
        for xi, v, e in zip(xs, m, sd):
            ax.text(xi, v + e + 0.015, fmt(v), ha="center", va="bottom", fontsize=11.5, color=INK,
                    fontweight="bold" if var == "pretrain" else "normal")
    for xi, d in zip(x, datasets):                     # heuristik berbeda per dataset: garis pendek per grup
        ax.hlines(heur1[d], xi - width - 0.04, xi + width + 0.04, color=INK_2, linestyle=(0, (4, 3)),
                  linewidth=1.4, zorder=5)
    ax.set_xticks(x, [f"{d}\n({n_test[d]} kasus uji)" for d in datasets], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [fmt(v) for v in np.arange(0, 1.01, 0.2)])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{mode}  ({sub})", loc="left", fontsize=13.5, color=NAVY, fontweight="bold", pad=10)
axes[0].set_ylabel("AC@1 (data uji)", fontsize=13)

handles, labels = axes[0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=INK_2, linestyle=(0, (4, 3)), linewidth=1.4))
labels.append("Heuristik (per dataset)")
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.905), ncol=3, frameon=False,
           fontsize=11.5, handlelength=1.6, columnspacing=1.6)
fig.suptitle("EX-1: pre-training vs acak di dataset yang sama (rata-rata ± SD, 3 seed)", x=0.06, y=0.975,
             ha="left", fontsize=15, color=NAVY, fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.99, top=0.77, bottom=0.14, wspace=0.08)
fig.savefig(OUT / "ex1_finelast_fineall.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------ tabel 1: EX-1 FineLast vs FineAll
def pm(mode, d, var):
    m, sd = stat.loc[(mode, d, var), "mean"], stat.loc[(mode, d, var), "std"]
    return f"{fmt(m)} ± {fmt(sd)}"


HEAD_BG, SUB_BG, ROW_ALT, LINE = NAVY, "#1f4a8a", "#f3f5f9", "#d5d9e0"
cols = [0.0, 0.20, 0.34, 0.505, 0.67, 0.835, 1.0]      # batas kolom (koordinat sumbu 0-1)
fig, ax = plt.subplots(figsize=(12, 4.0), dpi=200)
ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
row_h = 0.15
y_top = 0.98


def cell(x0, x1, y0, h, text, bg=None, color=INK, bold=False, size=14):
    if bg:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, h, facecolor=bg, edgecolor="white", linewidth=1.5))
    ax.text((x0 + x1) / 2, y0 + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            fontweight="bold" if bold else "normal")


# header dua tingkat
y1 = y_top - row_h
cell(cols[0], cols[1], y1 - row_h, 2 * row_h, "Dataset", HEAD_BG, "white", True)
cell(cols[1], cols[2], y1 - row_h, 2 * row_h, "Heuristik", HEAD_BG, "white", True)
cell(cols[2], cols[4], y1, row_h, "FineLast (encoder dibekukan)", HEAD_BG, "white", True)
cell(cols[4], cols[6], y1, row_h, "FineAll (encoder ikut dilatih)", HEAD_BG, "white", True)
for c0, c1, t in [(2, 3, "Pre-training"), (3, 4, "Acak"), (4, 5, "Pre-training"), (5, 6, "Acak")]:
    cell(cols[c0], cols[c1], y1 - row_h, row_h, t, SUB_BG, "white", True, 13)

# isi
y = y1 - row_h
for k, d in enumerate(datasets):
    y -= row_h
    bg = ROW_ALT if k % 2 else "white"
    ax.add_patch(plt.Rectangle((0, y), 1, row_h, facecolor=bg, edgecolor="none"))
    ax.text(cols[0] + 0.015, y + row_h / 2, d, ha="left", va="center", fontsize=14, color=INK,
            fontweight="bold")
    ax.text(cols[0] + 0.075, y + row_h / 2, f"({n_test[d]} kasus uji)", ha="left", va="center",
            fontsize=12, color=INK_2)
    cell(cols[1], cols[2], y, row_h, fmt(heur1[d]))
    for c0, mode, var in [(2, "FineLast", "pretrain"), (3, "FineLast", "scratch"),
                          (4, "FineAll", "pretrain"), (5, "FineAll", "scratch")]:
        cell(cols[c0], cols[c0 + 1], y, row_h, pm(mode, d, var))
    ax.plot([0, 1], [y, y], color=LINE, linewidth=1)

ax.text(0, y - 0.04, "AC@1 pada data uji, rata-rata ± SD dari 3 seed. Pre-training dan pengujian di dataset "
        "yang sama.\nSelisih pre-training vs acak paling besar 0,03 (≈ 1–2 kasus uji) di semua sel.",
        ha="left", va="top", fontsize=11.5, color=INK_2, linespacing=1.5)
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
fig.savefig(OUT / "ex1_tabel_finelast_fineall.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------ gambar 4: uji "anomali dihapus"
# Kasus uji nyata (RE1, Online Boutique, gangguan CPU di adservice). Penggantian segmen sama persis
# dengan remove_anomaly() di scripts/diagnose_prior.py.
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from ssl_rcl.data import CORE_METRICS, build_samples  # noqa: E402

CASE, METRIC = "re1ob_adservice_cpu_5", "cpu"
case = next(c for c in build_samples(ROOT / "dataset", ["RE1"], cache_dir=ROOT / "cache", verbose=False)
            if c["name"] == CASE)
off, T = case["inject_offset"], case["x"].shape[1]
post_len = T - off
k, root = CORE_METRICS.index(METRIC), case["label"]
orig = case["x"][:, :, k]
removed = orig.copy()
removed[:, off:] = orig[:, off - post_len:off]
t = np.arange(T) - off                                  # detik relatif terhadap waktu gangguan
others = [i for i in range(orig.shape[0]) if i != root and case["mask"][i, k]]

PRE_BG, POST_BG, ROOT_C, OTHER_C = "#eef2f8", "#fdeee6", "#2a78d6", "#c9c8c2"
fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), dpi=200, sharex=True)
ylim = (min(orig.min(), -3) - 0.5, orig[root].max() + 2.5)
for ax, data, title, post_label in [
        (axes[0], orig, "Data asli", "240 detik sesudah gangguan"),
        (axes[1], removed, "Anomali dihapus", "diganti data normal dari kasus yang sama")]:
    ax.axvspan(-off, 0, color=PRE_BG, zorder=0)
    ax.axvspan(0, post_len, color=POST_BG if data is orig else PRE_BG, zorder=0)
    for i in others:
        ax.plot(t, data[i], color=OTHER_C, linewidth=1.0, zorder=2)
    ax.plot(t, data[root], color=ROOT_C, linewidth=2.0, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.4, zorder=4)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylabel("CPU (z-score)", fontsize=12)
    ax.text(-off + 6, ylim[1] - 0.6, "300 detik sebelum gangguan (normal)", ha="left", va="top",
            fontsize=12, color=NAVY, fontweight="bold")
    ax.text(6, ylim[1] - 0.6, post_label, ha="left", va="top", fontsize=12, color=NAVY, fontweight="bold")
    ax.text(-off - 18, ylim[1] + 0.4, title, ha="left", va="bottom", fontsize=15, color=NAVY,
            fontweight="bold")

axes[0].annotate("gangguan disuntikkan", xy=(0, 8.0), xytext=(-150, 8.0),
                 fontsize=11.5, color=INK, va="center", ha="center",
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.2))
peak = int(np.argmax(orig[root, off:])) + off
axes[0].annotate("CPU adservice melonjak", xy=(t[peak], orig[root, peak]), xytext=(150, orig[root].max() * 0.55),
                 fontsize=11.5, color=INK, ha="center",
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.2))
# panel bawah: tunjukkan asal potongan pengganti
src0 = -post_len
axes[1].hlines(ylim[0] + 0.35, src0, 0, color=INK_2, linewidth=2.5)
axes[1].hlines(ylim[0] + 0.35, 0, post_len, color=INK_2, linewidth=2.5, linestyle=(0, (3, 2)))
axes[1].text(src0 / 2, ylim[0] + 0.8, "potongan normal ini (240 detik) ...", ha="center", va="bottom",
             fontsize=11.5, color=INK, fontweight="bold",
             bbox=dict(facecolor=PRE_BG, edgecolor="none", pad=1.5))
axes[1].text(post_len / 2, ylim[0] + 0.8, "... disalin ke sini", ha="center", va="bottom",
             fontsize=11.5, color=INK, fontweight="bold",
             bbox=dict(facecolor=PRE_BG, edgecolor="none", pad=1.5))
axes[1].set_xlabel("Detik relatif terhadap waktu gangguan (0 = gangguan disuntikkan)", fontsize=12)
axes[1].set_xlim(-off, post_len)
axes[1].set_xticks(np.arange(-300, post_len + 1, 60))

from matplotlib.lines import Line2D  # noqa: E402

fig.legend([Line2D([], [], color=ROOT_C, linewidth=2.0), Line2D([], [], color=OTHER_C, linewidth=1.2)],
           ["Service sumber gangguan (adservice)", f"{len(others)} service lain"],
           loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=2, frameon=False, fontsize=11.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10, hspace=0.32)
fig.savefig(OUT / "uji_anomali_dihapus.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------ gambar 5: skema "anomali dihapus"
NORM_BG, NORM_EDGE, ANOM_BG, ANOM_EDGE = "#dde8f7", "#2a78d6", "#fbe0d3", "#eb6834"
PRE_W, INJ_W, POST_W = 300 / 540 * 0.80, 0.045, 240 / 540 * 0.80   # lebar blok sebanding durasi
X0 = 0.075
rng = np.random.default_rng(3)


def block(ax, x, y, w, h, bg, edge, title, sub, wave):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=edge, linewidth=2))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=15, color=INK, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.115, sub, ha="center", va="top", fontsize=11.5, color=INK_2)
    xs = np.linspace(x + 0.02, x + w - 0.02, 160)        # garis metrik bergaya sketsa
    base = y + 0.035
    if wave == "normal":
        ys = base + 0.02 + 0.010 * np.sin(np.linspace(0, 14, 160)) + rng.normal(0, 0.004, 160)
    else:
        ramp = np.clip(np.linspace(0, 6, 160), 0, 1)
        ys = base + 0.02 + 0.075 * ramp + rng.normal(0, 0.005, 160)
    ax.plot(xs, ys, color=edge, linewidth=2)


def row(ax, y, h, label, post_title, post_sub, post_kind):
    ax.text(X0 - 0.012, y + h / 2, label, ha="right", va="center", fontsize=14, color=NAVY, fontweight="bold")
    block(ax, X0, y, PRE_W, h, NORM_BG, NORM_EDGE, "300 detik normal", "sebelum gangguan", "normal")
    xi = X0 + PRE_W
    ax.add_patch(plt.Rectangle((xi, y), INJ_W, h, facecolor=NAVY, edgecolor=NAVY))
    ax.text(xi + INJ_W / 2, y + h / 2, "INJEKSI", rotation=90, ha="center", va="center", fontsize=12,
            color="white", fontweight="bold")
    bg, edge = (ANOM_BG, ANOM_EDGE) if post_kind == "anomali" else (NORM_BG, NORM_EDGE)
    block(ax, xi + INJ_W, y, POST_W, h, bg, edge, post_title, post_sub, post_kind)


fig, ax = plt.subplots(figsize=(12, 5.4), dpi=200)
ax.set_axis_off(); ax.set_xlim(-0.12, 0.93); ax.set_ylim(-0.03, 1)
h = 0.30
row(ax, 0.62, h, "Data asli", "240 detik anomali", "metrik melonjak", "anomali")
row(ax, 0.12, h, "Anomali\ndihapus", "240 detik normal", "salinan data sebelum gangguan", "normal")

# panah: dari 240 detik terakhir sebelum injeksi (baris bawah) ke blok pengganti
src_x0, src_x1 = X0 + PRE_W - POST_W, X0 + PRE_W
dst_x0 = X0 + PRE_W + INJ_W
ax.hlines(0.085, src_x0, src_x1, color=INK_2, linewidth=3)
ax.text((src_x0 + src_x1) / 2, 0.06, "240 detik terakhir sebelum gangguan", ha="center", va="top",
        fontsize=11, color=INK_2)
ax.annotate("", xy=(dst_x0 + POST_W / 2, 0.095), xytext=(src_x1 - 0.03, 0.07),
            arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.8, connectionstyle="arc3,rad=0.35",
                            shrinkA=2, shrinkB=2))
ax.text(dst_x0 + POST_W / 2 + 0.02, 0.06, "disalin ke sini", ha="left", va="center", fontsize=12,
        color=INK, fontweight="bold")
# perubahan antar baris
ax.annotate("", xy=(X0 + PRE_W + INJ_W + POST_W / 2, 0.12 + h + 0.005),
            xytext=(X0 + PRE_W + INJ_W + POST_W / 2, 0.62 - 0.005),
            arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.8))
ax.text(X0 + PRE_W + INJ_W + POST_W / 2 + 0.015, 0.52, "diganti", ha="left", va="center", fontsize=12,
        color=INK, fontweight="bold")
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
fig.savefig(OUT / "skema_anomali_dihapus.png", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------ gambar 6: langkah berikutnya
STEPS = [
    ("Ulangi EX-1 dan EX-2", "dengan TimeMAE dan RUN."),
    ("Tambahkan varian PCA", "dan dataset ART."),
]
BOX_EDGE = "#7f93b8"
fig, ax = plt.subplots(figsize=(12, 4.6), dpi=200)
ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
num_x, num_w, txt_x, txt_w, box_h = 0.13, 0.085, 0.24, 0.70, 0.36
ys = [0.56, 0.08]
spine_x = 0.07
ax.plot([spine_x, spine_x], [ys[1] + box_h / 2, ys[0] + box_h / 2], color=BOX_EDGE, linewidth=2)
for i, ((line1, line2), y) in enumerate(zip(STEPS, ys), start=1):
    ax.plot([spine_x, num_x], [y + box_h / 2] * 2, color=BOX_EDGE, linewidth=2)
    ax.add_patch(plt.Rectangle((num_x, y), num_w, box_h, facecolor=NAVY, edgecolor=NAVY))
    ax.text(num_x + num_w / 2, y + box_h / 2, str(i), ha="center", va="center", fontsize=44,
            color="white", fontweight="bold")
    ax.add_patch(plt.Rectangle((txt_x, y), txt_w, box_h, facecolor="white", edgecolor=BOX_EDGE, linewidth=2))
    ax.text(txt_x + 0.03, y + box_h * 0.63, line1, ha="left", va="center", fontsize=20, color=INK_2)
    ax.text(txt_x + 0.03, y + box_h * 0.30, line2, ha="left", va="center", fontsize=20, color=INK_2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
fig.savefig(OUT / "langkah_berikutnya.png", facecolor=SURFACE)
plt.close(fig)

print("Disimpan:", OUT / "ex2_hasil_re2.png", "|", OUT / "ex2_uji_bootstrap.png", "|",
      OUT / "ex1_finelast_fineall.png", "|", OUT / "ex1_tabel_finelast_fineall.png", "|",
      OUT / "uji_anomali_dihapus.png", "|", OUT / "skema_anomali_dihapus.png", "|",
      OUT / "langkah_berikutnya.png")
