"""TimeMAE (Cheng et al., WSDM 2026), implementasi ulang mengikuti paper di reference/TimeMAE.pdf.

Antarmuka encoder sama seperti `ts2vec.TSEncoder`: `encoder(x[B, T, C], mask="all_true") -> [B, T, D]`,
sehingga `finetune.service_features` dan `run_ex1.py` bisa dipakai tanpa perubahan.

Komponen:
- Window slicing: series dipotong menjadi patch non-overlap sepanjang `patch_size`; tiap patch
  di-embed oleh 1-D CNN (kernel = stride = patch_size) lintas kanal.
- Online encoder H: embedding + posisi -> transformer encoder. Saat pre-training hanya patch
  VISIBLE yang masuk (token mask tidak pernah masuk encoder).
- Decoupled encoder F: query = token mask (+ posisi), key/value = keluaran H di posisi visible
  (cross-attention). Hanya memprediksi posisi yang di-mask.
- Target encoder: salinan online encoder yang di-update momentum (EMA), melihat seluruh patch.
- MRR: MSE antara keluaran F di posisi mask dan keluaran target encoder (stop-gradient).
- MCC: tokenizer (codebook) memberi codeword pada tiap patch lewat inner product + relaxed
  (Gumbel) softmax dengan straight-through estimator; F memprediksi codeword patch yang di-mask.
- Loss total = alpha * MCC + beta * MRR + gamma * keragaman. Suku keragaman (entropi pemakaian codeword)
  BUKAN dari paper: tanpa itu tokenizer kolaps ke satu codeword pada uji awal (perplexity 1/128).

Representasi per-timestamp untuk fine-tuning: keluaran H per patch diulang sepanjang `patch_size`
timestep, sehingga max-pool pra/pasca-injeksi di `service_features` tetap berlaku. Karena itu
`patch_size` harus membagi `inject_offset` (300) dan T (540); default 12.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- encoder
class TimeMAEEncoder(nn.Module):
    """x [B, T, input_dims] -> representasi [B, T, output_dims] (per patch, diulang per timestep)."""

    def __init__(self, input_dims, output_dims=64, patch_size=12, depth=8, n_heads=4, ff_mult=4,
                 dropout=0.2, max_patches=256):
        super().__init__()
        self.input_dims, self.output_dims, self.patch_size = input_dims, output_dims, patch_size
        self.depth, self.n_heads, self.ff_mult, self.dropout_p, self.max_patches = depth, n_heads, ff_mult, dropout, max_patches
        self.patch_embed = nn.Conv1d(input_dims, output_dims, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.randn(1, max_patches, output_dims) * 0.02)
        layer = nn.TransformerEncoderLayer(output_dims, n_heads, ff_mult * output_dims, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(output_dims)

    def config(self) -> dict:
        return dict(input_dims=self.input_dims, output_dims=self.output_dims, patch_size=self.patch_size,
                    depth=self.depth, n_heads=self.n_heads, ff_mult=self.ff_mult, dropout=self.dropout_p,
                    max_patches=self.max_patches)

    def embed(self, x):
        """x [B, T, C] -> embedding patch [B, N, D] (belum + posisi). T dipadding nol ke kelipatan patch_size."""
        x = torch.nan_to_num(x)
        pad = (-x.size(1)) % self.patch_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return self.patch_embed(x.transpose(1, 2)).transpose(1, 2)

    def encode(self, z, keep_idx=None):
        """z [B, N, D] embedding patch. keep_idx [B, n] = patch yang ikut (None = semua). -> [B, n, D]."""
        pos = self.pos[:, :z.size(1)].expand(z.size(0), -1, -1)
        h = z + pos
        if keep_idx is not None:
            h = h.gather(1, keep_idx[..., None].expand(-1, -1, h.size(-1)))
        return self.norm(self.blocks(h))

    def forward(self, x, mask=None):                     # `mask` hanya demi kompatibilitas dengan TSEncoder
        T = x.size(1)
        h = self.encode(self.embed(x))                   # [B, N, D]
        return h.repeat_interleave(self.patch_size, dim=1)[:, :T]


# ---------------------------------------------------------------- pre-training model
class _CrossLayer(nn.Module):
    """Satu layer decoupled encoder: cross-attention (query = posisi mask, key/value = visible) + FFN."""

    def __init__(self, d, n_heads, ff_mult, dropout):
        super().__init__()
        self.n1, self.nkv, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        kv = self.nkv(kv)
        q = q + self.drop(self.attn(self.n1(q), kv, kv, need_weights=False)[0])
        return q + self.drop(self.ff(self.n2(q)))


class TimeMAE(nn.Module):
    def __init__(self, encoder: TimeMAEEncoder, vocab_size=128, decoupled_depth=6, mask_ratio=0.6,
                 momentum=0.99, tau=1.0):
        super().__init__()
        d = encoder.output_dims
        self.encoder = encoder                                           # online encoder H
        self.target = copy.deepcopy(encoder).eval()                      # target encoder (EMA)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.decoder = nn.ModuleList([_CrossLayer(d, encoder.n_heads, encoder.ff_mult, encoder.dropout_p)
                                      for _ in range(decoupled_depth)])
        self.dec_norm = nn.LayerNorm(d)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.codebook = nn.Parameter(torch.randn(vocab_size, d) * 0.02)
        self.vocab_size, self.mask_ratio, self.momentum, self.tau = vocab_size, mask_ratio, momentum, tau

    @torch.no_grad()
    def update_target(self):
        for pt, po in zip(self.target.parameters(), self.encoder.parameters()):
            pt.mul_(self.momentum).add_(po.detach(), alpha=1 - self.momentum)
        for bt, bo in zip(self.target.buffers(), self.encoder.buffers()):
            bt.copy_(bo)

    def forward(self, x):
        """x [B, T, C] -> (loss_mcc, loss_mrr, loss_div, perplexity, rep_std). Mask patch acak, jumlahnya sama untuk tiap sampel."""
        enc = self.encoder
        z = enc.embed(x)                                                 # [B, N, D]
        B, N, D = z.shape
        n_mask = min(max(int(round(self.mask_ratio * N)), 1), N - 1)
        order = torch.rand(B, N, device=z.device).argsort(dim=1)
        mask_idx, vis_idx = order[:, :n_mask], order[:, n_mask:]

        # --- online: encoder H pada patch visible, lalu decoupled encoder F pada posisi mask
        h_vis = enc.encode(z, vis_idx)                                   # [B, n_vis, D]
        pos_m = enc.pos[:, :N].expand(B, -1, -1).gather(1, mask_idx[..., None].expand(-1, -1, D))
        q = self.mask_token.expand(B, n_mask, -1) + pos_m
        for layer in self.decoder:
            q = layer(q, h_vis)
        y = self.dec_norm(q)                                             # [B, n_mask, D]

        # --- MRR: regresi ke target encoder (seluruh patch, tanpa gradien)
        with torch.no_grad():
            t_full = self.target.encode(self.target.embed(x))            # [B, N, D]
            t_mask = t_full.gather(1, mask_idx[..., None].expand(-1, -1, D))
        loss_mrr = F.mse_loss(y, t_mask)

        # --- MCC: codeword patch yang di-mask (inner product + Gumbel softmax + STE)
        z_mask = F.layer_norm(z.detach().gather(1, mask_idx[..., None].expand(-1, -1, D)), (D,))
        tok_logits = z_mask @ self.codebook.t()                          # [B, n_mask, V]
        assign = F.gumbel_softmax(tok_logits, tau=self.tau, hard=True, dim=-1)
        log_p = F.log_softmax(y @ self.codebook.t(), dim=-1)
        loss_mcc = -(assign * log_p).sum(-1).mean()

        # Tanpa regularisasi, tokenizer kolaps ke 1 codeword (tugas MCC jadi trivial). Loss keragaman
        # = log V - H(rata-rata distribusi codeword); 0 bila seluruh codeword terpakai merata.
        p_bar = F.softmax(tok_logits, dim=-1).mean(dim=(0, 1))
        loss_div = math.log(self.vocab_size) + (p_bar * (p_bar + 1e-9).log()).sum()

        with torch.no_grad():                                            # diagnostik: pemakaian codeword & kolaps
            used = tok_logits.argmax(-1).flatten()
            p = torch.bincount(used, minlength=self.vocab_size).float()
            p = p / p.sum()
            perplexity = torch.exp(-(p[p > 0] * p[p > 0].log()).sum())
            rep_std = t_full.std(dim=(0, 1)).mean()                      # -> 0 berarti representasi kolaps
        return loss_mcc, loss_mrr, loss_div, perplexity, rep_std


# ---------------------------------------------------------------- pre-training
def pretrain_timemae(windows: np.ndarray, input_dims: int, device, n_iters: int = 3000, batch_size: int = 64,
                     lr: float = 1e-3, output_dims: int = 64, patch_size: int = 12, depth: int = 8,
                     n_heads: int = 4, decoupled_depth: int = 6, vocab_size: int = 128,
                     mask_ratio: float = 0.6, momentum: float = 0.99, alpha: float = 1.0, beta: float = 1.0,
                     gamma: float = 5.0,
                     dropout: float = 0.2, grad_clip: float = 1.0, seed: int = 42, log_every: int = 100,
                     verbose=True):
    """Pre-training self-supervised pada window [M, T, input_dims] (tanpa label).

    Mengembalikan (encoder_online, log); `log` = total loss per iterasi. Sama seperti `pretrain_ts2vec`,
    jumlah langkah ditentukan `n_iters` (bukan epoch) supaya biaya terkontrol.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    enc = TimeMAEEncoder(input_dims, output_dims, patch_size, depth, n_heads, dropout=dropout)
    model = TimeMAE(enc, vocab_size, decoupled_depth, mask_ratio, momentum).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)

    data = torch.from_numpy(windows).float()
    M = data.shape[0]
    if M < batch_size:
        raise ValueError(f"window pre-training ({M}) lebih sedikit dari batch_size ({batch_size})")
    hist, it, t0 = [], 0, time.time()
    model.train(); model.target.eval()
    while it < n_iters:
        perm = torch.randperm(M, generator=g)
        for b in range(0, M - batch_size + 1, batch_size):
            if it >= n_iters:
                break
            x = data[perm[b:b + batch_size]].to(device, non_blocking=True)
            l_mcc, l_mrr, l_div, ppl, rstd = model(x)
            loss = alpha * l_mcc + beta * l_mrr + gamma * l_div
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            model.update_target()
            hist.append((loss.item(), l_mcc.item(), l_mrr.item(), ppl.item(), rstd.item(), l_div.item()))
            it += 1
            if verbose and (it % log_every == 0 or it == n_iters):
                a = np.mean(hist[-log_every:], axis=0)
                print(f"  iter {it:5d}/{n_iters}  loss {a[0]:.4f} (MCC {a[1]:.3f} MRR {a[2]:.4f} div {a[5]:.2f})  "
                      f"codeword-ppl {a[3]:.1f}/{vocab_size}  std-repr {a[4]:.3f}  ({time.time() - t0:.0f}s)")
    encoder = copy.deepcopy(model.encoder).eval()
    return encoder, [h[0] for h in hist]
