"""Loader dan preprocessing metrik RCAEval (RE1/RE2/RE3).

Skema (lihat report/skema_fitur_metrik.md):
- Satu kasus = tensor [N_service, T, K] dengan K jenis metrik tetap + mask [N_service, K].
- Encoder melihat per service [T, K], sehingga input_dims = K sama di semua sistem/RE.
- Label = indeks root-cause service di antara service kasus tersebut (lokalisasi).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CORE_METRICS: tuple[str, ...] = ("cpu", "mem", "latency-50", "latency-90", "workload")
EXTENDED_METRICS: tuple[str, ...] = CORE_METRICS + ("diskio", "socket")

# Kasus rusak (lihat report/skema_fitur_metrik.md, bagian "Kasus bermasalah").
EXCLUDED_CASES = {
    "re1ob_currencyservice_loss_1",       # inject_time.txt terpotong (16933142)
    "re1ob_productcatalogservice_cpu_3",  # hanya 62 detik data, injeksi di luar rentang
}

# Kontainer/pod yang bukan service aplikasi.
NON_APP_SERVICES = {
    "main", "istio-init", "istio-proxy", "frontend-check", "frontend-external",
    "PassthroughCluster", "InboundPassthroughClusterIpv4",
}

CASE_RE = re.compile(r"^re(?P<re>\d)(?P<sys>ob|ss|tt)_(?P<svc>.+)_(?P<fault>[a-z0-9]+)_(?P<rep>\d+)$")


@dataclass(frozen=True)
class CaseMeta:
    name: str
    re: str            # "RE1" / "RE2" / "RE3"
    system: str        # "ob" / "ss" / "tt"
    root_service: str
    fault: str
    rep: int
    path: Path


def list_cases(data_root: str | Path, re_names=("RE1", "RE2", "RE3"),
               exclude=EXCLUDED_CASES) -> list[CaseMeta]:
    """Daftar kasus valid (punya metrics.parquet dan inject_time.txt), terurut nama."""
    out = []
    for re_name in re_names:
        for d in sorted((Path(data_root) / re_name).iterdir()):
            m = CASE_RE.match(d.name)
            if (not m or d.name in exclude or not (d / "metrics.parquet").exists()
                    or not (d / "inject_time.txt").exists()):
                continue
            out.append(CaseMeta(d.name, re_name, m["sys"], m["svc"], m["fault"], int(m["rep"]), d))
    return out


def _split_col(col: str) -> tuple[str, str]:
    svc, _, metric = col.rpartition("_")
    return svc, metric


def select_services(columns, root_service: str | None = None) -> list[str]:
    """Service aplikasi = punya kolom `cpu` dan bukan kontainer infrastruktur. Terurut alfabet."""
    svcs = {s for s, m in map(_split_col, columns) if m == "cpu" and s not in NON_APP_SERVICES}
    if root_service is not None and root_service not in svcs:
        raise ValueError(f"root service {root_service!r} tidak punya kolom cpu")
    return sorted(svcs)


def _robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median dan skala (MAD*1.4826, fallback std, fallback 1) per kolom; x [T, C] boleh NaN."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(x, axis=0)
            mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
            std = np.nanstd(x, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return np.nan_to_num(med), np.nan_to_num(scale, nan=1.0)


def load_case(meta: CaseMeta, metrics=CORE_METRICS, pre: int | None = 300, post: int | None = 240,
              clip: float = 10.0) -> dict:
    """Muat dan praproses satu kasus.

    pre/post : panjang window (detik) sebelum/sesudah injeksi. None = ambil seluruh series
               pada sisi tersebut (dipakai untuk pre-training).
    Normalisasi: robust z-score per series (service x metrik) dengan statistik dari
    segmen pra-injeksi seluruh rekaman, lalu di-clip ke [-clip, clip].
    """
    df = pd.read_parquet(meta.path / "metrics.parquet")
    inject = int((meta.path / "inject_time.txt").read_text().strip())
    df = df.drop_duplicates("time").set_index("time").sort_index()

    t0, t1 = int(df.index.min()), int(df.index.max())
    if not (t0 < inject <= t1):
        raise ValueError(f"{meta.name}: inject_time di luar rentang data")

    services = select_services(df.columns, meta.root_service)
    K, N = len(metrics), len(services)
    cols = [f"{s}_{m}" for s in services for m in metrics]
    present = np.array([c in df.columns for c in cols]).reshape(N, K)

    # Grid 1 detik penuh, kolom yang tidak ada diisi NaN.
    full = df.reindex(index=np.arange(t0, t1 + 1), columns=cols).astype(np.float64)
    full = full.ffill().bfill()
    values = full.to_numpy()                                   # [T_full, N*K]
    all_nan = np.isnan(values).all(axis=0).reshape(N, K)
    mask = present & ~all_nan

    pre_seg = values[: inject - t0]
    med, scale = _robust_stats(pre_seg)
    z = np.clip((values - med) / scale, -clip, clip)
    z = np.nan_to_num(z)
    z[:, ~mask.reshape(-1)] = 0.0

    start = t0 if pre is None else inject - pre
    end = t1 + 1 if post is None else inject + post
    if start < t0:
        raise ValueError(f"{meta.name}: data pra-injeksi kurang dari {pre} detik")
    # Rekaman yang berhenti lebih awal (mis. re2ob_checkoutservice_cpu_2, 209 s pasca-injeksi)
    # dipadding dengan nilai terakhir; jumlahnya dicatat di `pad_post`.
    pad_post = max(0, end - (t1 + 1))
    z = z[start - t0: end - t0]                                # [T, N*K]
    if pad_post:
        z = np.concatenate([z, np.repeat(z[-1:], pad_post, axis=0)])

    x = z.reshape(len(z), N, K).transpose(1, 0, 2).astype(np.float32)  # [N, T, K]
    return dict(
        name=meta.name, re=meta.re, system=meta.system, fault=meta.fault,
        root_service=meta.root_service, services=services,
        label=services.index(meta.root_service),
        x=x, mask=mask.astype(np.float32), inject_offset=inject - start, pad_post=pad_post,
    )


def build_samples(data_root, re_names, metrics=CORE_METRICS, pre=300, post=240,
                  cache_dir: str | Path | None = None, verbose=True) -> list[dict]:
    """Muat semua kasus pada re_names, dengan cache .pt per (RE, metrik, window)."""
    import torch

    samples = []
    for re_name in re_names:
        tag = f"{re_name}_K{len(metrics)}_pre{pre}_post{post}.pt"
        cache = Path(cache_dir) / tag if cache_dir else None
        if cache and cache.exists():
            part = torch.load(cache, weights_only=False)
        else:
            part = [load_case(m, metrics, pre, post) for m in list_cases(data_root, [re_name])]
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(part, cache)
        if verbose:
            print(f"{re_name}: {len(part)} kasus")
        samples += part
    return samples


def stratified_split(samples: list[dict], ratios=(0.6, 0.2, 0.2), seed: int = 42):
    """Split train/val/test terstratifikasi per (system, root_service). Mengembalikan indeks."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault((s["system"], s["root_service"]), []).append(i)
    splits = ([], [], [])
    for key in sorted(groups):
        idx = rng.permutation(groups[key])
        n = len(idx)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        splits[0].extend(idx[:n_tr]); splits[1].extend(idx[n_tr:n_tr + n_va]); splits[2].extend(idx[n_tr + n_va:])
    return tuple(sorted(int(i) for i in s) for s in splits)


def make_pretrain_windows(samples: list[dict], length: int, stride: int, with_mask: bool = False) -> np.ndarray:
    """Potong series per service menjadi window [M, length, K] untuk pre-training SSL (tanpa label).

    Kanal yang di-mask tetap 0. with_mask=True menambahkan mask kanal sebagai K kanal tambahan
    (konstan sepanjang waktu) -> [M, length, 2K], sama dengan input encoder saat fine-tuning.
    Gunakan hanya sampel split train agar tidak ada kebocoran.
    """
    out = []
    for s in samples:
        x = s["x"]                                    # [N, T, K]
        if with_mask:
            m = np.broadcast_to(s["mask"][:, None, :], x.shape)
            x = np.concatenate([x, m], axis=-1).astype(np.float32)
        T = x.shape[1]
        for st in range(0, max(T - length, 0) + 1, stride):
            w = x[:, st:st + length]
            if w.shape[1] == length:
                out.append(w)
    return np.concatenate(out, axis=0) if out else np.zeros((0, length, samples[0]["x"].shape[2]), np.float32)


class CaseDataset:
    """Dataset PyTorch level kasus untuk fine-tuning lokalisasi root cause."""

    def __init__(self, samples: list[dict], indices=None):
        self.samples = samples if indices is None else [samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_cases(batch: list[dict]) -> dict:
    """Padding dimensi service (N berbeda antar sistem). service_mask=1 untuk service asli."""
    import torch

    n_max = max(b["x"].shape[0] for b in batch)
    T, K = batch[0]["x"].shape[1:]
    if any(b["x"].shape[1:] != (T, K) for b in batch):
        raise ValueError("Panjang window/K harus sama dalam satu batch")
    x = torch.zeros(len(batch), n_max, T, K)
    ch_mask = torch.zeros(len(batch), n_max, K)
    svc_mask = torch.zeros(len(batch), n_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["x"].shape[0]
        x[i, :n] = torch.from_numpy(b["x"])
        ch_mask[i, :n] = torch.from_numpy(b["mask"])
        svc_mask[i, :n] = True
    return dict(
        x=x, channel_mask=ch_mask, service_mask=svc_mask,
        label=torch.tensor([b["label"] for b in batch]),
        name=[b["name"] for b in batch], system=[b["system"] for b in batch],
    )


def case_meta_table(samples: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: s[k] for k in ("name", "re", "system", "fault", "root_service", "label")}
                         | {"n_services": len(s["services"]), "T": s["x"].shape[1],
                            "masked_channels": int((s["mask"] == 0).sum())} for s in samples])
