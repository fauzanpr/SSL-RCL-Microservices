"""RUN (Lin et al., AAAI 2024, "Root Cause Analysis in Microservice Using Neural Granger Causal Discovery"),
diimplementasikan ulang untuk tahap pre-training (paper Bagian 4.1, `reference/RUN.pdf`).

Yang dipakai dari RUN hanya TAHAP PRE-TRAINING: backbone DLinear + kontras tanpa pasangan negatif.
Tahap neural Granger causal discovery dan PageRank (Bagian 4.2-4.3) tidak dipakai, karena tugas di
proyek ini adalah fine-tuning lokalisasi service di atas encoder (`finetune.py`), bukan graf kausal.

Antarmuka encoder sama seperti `ts2vec.TSEncoder`: `encoder(x[B, T, C], mask="all_true") -> [B, T, D]`,
sehingga `finetune.service_features` dan `run_ex1.py` bisa dipakai tanpa perubahan.

Komponen:
- RUNEncoder (backbone DLinear, Zeng et al. 2023): dekomposisi deret (moving average = trend, sisanya = seasonal),
  lalu satu transformasi linear atas jendela geser sepanjang `window` (paper: w = 32) untuk masing-masing komponen.
  Jendela bersifat kausal (timestamp t hanya melihat t-w+1..t), sehingga representasi pasca-injeksi tidak
  "mengintip" ke masa depan dan panjang input boleh berubah-ubah (random crop).
- Kontras tanpa negatif (paper Eq. 1-2, skema SimSiam): dua crop yang overlap; timestamp yang SAMA dengan konteks
  berbeda adalah pasangan positif, tidak ada pasangan negatif. Loss
      L_con = -1/2 (cos(z1, sg(p2)) + cos(sg(p1), z2)),  p_i = f(y_i),  z_i = g(f(y_i))
  dengan g = projector/predictor MLP dan sg = stop-gradient, dihitung per timestamp pada bagian overlap.
- Pembeda konteks: TS2Vec membuat dua view berbeda lewat timestamp masking (binomial p=0.5) di dalam encoder. Paper RUN
  hanya menyebut random crop "mengikuti Yue et al. 2022", padahal receptive field DLinear pendek (w=32), jadi tanpa
  masking dua crop menghasilkan representasi identik di sebagian besar overlap dan loss menjadi trivial. Maka masking
  timestamp yang sama seperti TS2Vec dipakai di sini sebagai sumber perbedaan konteks (saat training saja).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts2vec import take_per_row


# ---------------------------------------------------------------- encoder
class _MovingAvg(nn.Module):
    """Moving average sepanjang waktu (padding replikasi di kedua ujung, panjang output = input)."""

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):                                           # x [B, C, T]
        if self.kernel <= 1:
            return x
        left = (self.kernel - 1) // 2
        right = self.kernel - 1 - left
        x = torch.cat([x[..., :1].expand(-1, -1, left), x, x[..., -1:].expand(-1, -1, right)], dim=-1)
        return F.avg_pool1d(x, self.kernel, stride=1)


class RUNEncoder(nn.Module):
    """x [B, T, input_dims] -> representasi [B, T, output_dims]."""

    def __init__(self, input_dims, output_dims=64, window=32, ma_kernel=25, mask_p=0.5):
        super().__init__()
        self.input_dims, self.output_dims, self.window, self.ma_kernel, self.mask_p = \
            input_dims, output_dims, window, ma_kernel, mask_p
        self.decomp = _MovingAvg(ma_kernel)
        # "Linear atas jendela w" = Conv1d dengan kernel w (bobot dibagi di sepanjang waktu), satu untuk tiap komponen.
        self.trend = nn.Conv1d(input_dims, output_dims, window)
        self.seasonal = nn.Conv1d(input_dims, output_dims, window)

    def config(self) -> dict:
        return dict(input_dims=self.input_dims, output_dims=self.output_dims, window=self.window,
                    ma_kernel=self.ma_kernel, mask_p=self.mask_p)

    def forward(self, x, mask=None):
        x = torch.nan_to_num(x)                                     # [B, T, C]
        if mask is None:
            mask = "binomial" if self.training else "all_true"
        if mask == "binomial":                                      # konteks berbeda antar view (lihat docstring modul)
            m = torch.bernoulli(torch.full(x.shape[:2], 1 - self.mask_p, device=x.device)).bool()
            x = x * m.unsqueeze(-1)
        x = x.transpose(1, 2)                                       # [B, C, T]
        trend = self.decomp(x)
        seasonal = x - trend
        pad = lambda v: torch.cat([v[..., :1].expand(-1, -1, self.window - 1), v], dim=-1)  # noqa: E731  kausal
        out = self.trend(pad(trend)) + self.seasonal(pad(seasonal))  # [B, D, T]
        return out.transpose(1, 2)


class _Projector(nn.Module):
    """g pada Eq. 2: MLP kecil (Linear-LayerNorm-ReLU-Linear) yang memetakan representasi ke ruang kontras."""

    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or d
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


def run_contrastive_loss(p1, p2, z1, z2):
    """Eq. 2. p_i = f(y_i) (target, di-stop-gradient), z_i = g(p_i). Semua [B, L, D] pada timestamp overlap."""
    l1 = F.cosine_similarity(z1, p2.detach(), dim=-1).mean()
    l2 = F.cosine_similarity(z2, p1.detach(), dim=-1).mean()
    return -0.5 * (l1 + l2)


# ---------------------------------------------------------------- pre-training
def pretrain_run(windows: np.ndarray, input_dims: int, device, n_iters: int = 2000, batch_size: int = 128,
                 lr: float = 1e-3, output_dims: int = 64, window: int = 32, ma_kernel: int = 25,
                 mask_p: float = 0.5, min_crop: int = 64, seed: int = 42, log_every: int = 100, verbose=True):
    """Pre-training self-supervised pada window [M, T, input_dims] (tanpa label).

    Mengembalikan (encoder, log); `log` = loss per iterasi. Jumlah langkah ditentukan `n_iters` (bukan epoch),
    sama seperti `pretrain_ts2vec`/`pretrain_timemae`, supaya biaya terkontrol dan sebanding antar model.
    Default paper: Adam, lr 1e-3, batch 128, w = 32.
    """
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    enc = RUNEncoder(input_dims, output_dims, window, ma_kernel, mask_p).to(device)
    proj = _Projector(output_dims).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=lr)

    data = torch.from_numpy(windows).float()
    M, T = data.shape[:2]
    if M < batch_size:
        raise ValueError(f"window pre-training ({M}) lebih sedikit dari batch_size ({batch_size})")
    min_crop = min(max(min_crop, 2), T)
    hist, it, t0 = [], 0, time.time()
    enc.train(); proj.train()
    while it < n_iters:
        perm = torch.randperm(M, generator=g)
        for b in range(0, M - batch_size + 1, batch_size):
            if it >= n_iters:
                break
            x = data[perm[b:b + batch_size]].to(device, non_blocking=True)
            # random cropping dengan overlap (Eq. 1: a1 <= a2 <= b1 <= b2), logika sama seperti kode resmi TS2Vec
            crop_l = int(rng.integers(min_crop, T + 1))
            crop_left = int(rng.integers(T - crop_l + 1))
            crop_right = crop_left + crop_l
            crop_eleft = int(rng.integers(crop_left + 1))
            crop_eright = int(rng.integers(crop_right, T + 1))
            offset = torch.from_numpy(rng.integers(-crop_eleft, T - crop_eright + 1, size=x.size(0))).to(device)

            p1 = enc(take_per_row(x, offset + crop_eleft, crop_right - crop_eleft))[:, -crop_l:]    # [B, L, D]
            p2 = enc(take_per_row(x, offset + crop_left, crop_eright - crop_left))[:, :crop_l]
            loss = run_contrastive_loss(p1, p2, proj(p1), proj(p2))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():                    # diagnostik kolaps: std repr ter-normalisasi-L2 ~ 1/sqrt(D) bila sehat
                rstd = F.normalize(p1.detach().reshape(-1, output_dims), dim=-1).std(dim=0).mean()
            hist.append((loss.item(), rstd.item()))
            it += 1
            if verbose and (it % log_every == 0 or it == n_iters):
                a = np.mean(hist[-log_every:], axis=0)
                print(f"  iter {it:5d}/{n_iters}  loss {a[0]:.4f}  std-repr {a[1]:.3f} "
                      f"(sehat ~{1 / np.sqrt(output_dims):.3f}, 0 = kolaps)  ({time.time() - t0:.0f}s)")
    encoder = copy.deepcopy(enc).eval()
    return encoder, [h[0] for h in hist]
