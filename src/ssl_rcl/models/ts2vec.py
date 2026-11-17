"""TS2Vec (Yue et al., AAAI 2022), diimplementasikan ulang mengikuti kode resmi
github.com/zhihanyue/ts2vec (default: hidden 64, output 320, depth 10, batch 16, lr 1e-3).

Komponen:
- TSEncoder: input projection -> timestamp masking (binomial p=0.5 saat training)
  -> dilated CNN 10 blok residual (dilasi 2^l) -> representasi per timestamp.
- hierarchical_contrastive_loss: instance-wise + temporal contrastive pada beberapa skala waktu.
- Contextual consistency: dua random crop yang overlap; representasi pada bagian overlap
  harus konsisten.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- encoder
class SamePadConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        rf = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=rf // 2, dilation=dilation)
        self.remove = 1 if rf % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.remove] if self.remove else out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = SamePadConv(out_ch, out_ch, kernel_size, dilation)
        self.projector = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch or final else None

    def forward(self, x):
        res = x if self.projector is None else self.projector(x)
        x = self.conv2(F.gelu(self.conv1(F.gelu(x))))
        return x + res


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(channels[i - 1] if i > 0 else in_ch, channels[i], kernel_size,
                      dilation=2 ** i, final=(i == len(channels) - 1))
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


class TSEncoder(nn.Module):
    """x [B, T, input_dims] -> representasi [B, T, output_dims]."""

    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10):
        super().__init__()
        self.input_dims, self.output_dims = input_dims, output_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(hidden_dims, [hidden_dims] * depth + [output_dims], 3)
        self.repr_dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        nan_mask = ~x.isnan().any(dim=-1)
        x = torch.nan_to_num(x)
        x = self.input_fc(x)                                   # [B, T, H]
        if mask is None:
            mask = "binomial" if self.training else "all_true"
        if mask == "binomial":
            m = torch.bernoulli(torch.full(x.shape[:2], 0.5, device=x.device)).bool()
        else:
            m = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        m &= nan_mask
        x = x * m.unsqueeze(-1)
        x = self.repr_dropout(self.feature_extractor(x.transpose(1, 2)))
        return x.transpose(1, 2)


# ---------------------------------------------------------------- losses
def instance_contrastive_loss(z1, z2):
    B = z1.size(0)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)             # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))                   # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def temporal_contrastive_loss(z1, z2):
    T = z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)                             # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss, d = z1.new_tensor(0.0), 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit and 1 - alpha != 0:
            loss = loss + (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def take_per_row(x, offsets, length):
    idx = offsets[:, None] + torch.arange(length, device=x.device)[None, :]
    return x[torch.arange(x.size(0), device=x.device)[:, None], idx]


# ---------------------------------------------------------------- pre-training
def pretrain_ts2vec(windows: np.ndarray, input_dims: int, device, n_iters: int = 600, batch_size: int = 16,
                    lr: float = 1e-3, output_dims: int = 320, hidden_dims: int = 64, depth: int = 10,
                    temporal_unit: int = 0, seed: int = 42, log_every: int = 50, verbose=True):
    """Pre-training self-supervised pada window [M, T, input_dims] (tanpa label).

    Mengembalikan (encoder_swa, log). Seperti kode resmi, encoder akhir adalah rata-rata
    bobot (SWA/AveragedModel) sepanjang training.
    """
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    net = TSEncoder(input_dims, output_dims, hidden_dims, depth).to(device)
    swa = torch.optim.swa_utils.AveragedModel(net)
    swa.update_parameters(net)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    data = torch.from_numpy(windows).float()
    M, T = data.shape[:2]
    log, it, t_start = [], 0, time.time()
    while it < n_iters:
        perm = torch.randperm(M, generator=g)
        for b in range(0, M - batch_size + 1, batch_size):
            if it >= n_iters:
                break
            x = data[perm[b:b + batch_size]].to(device)
            # random cropping dengan overlap (contextual consistency)
            crop_l = int(rng.integers(2 ** (temporal_unit + 1), T + 1))
            crop_left = int(rng.integers(T - crop_l + 1))
            crop_right = crop_left + crop_l
            crop_eleft = int(rng.integers(crop_left + 1))
            crop_eright = int(rng.integers(crop_right, T + 1))
            offset = torch.from_numpy(rng.integers(-crop_eleft, T - crop_eright + 1, size=x.size(0))).to(device)

            opt.zero_grad()
            out1 = net(take_per_row(x, offset + crop_eleft, crop_right - crop_eleft))[:, -crop_l:]
            out2 = net(take_per_row(x, offset + crop_left, crop_eright - crop_left))[:, :crop_l]
            loss = hierarchical_contrastive_loss(out1, out2, temporal_unit=temporal_unit)
            loss.backward()
            opt.step()
            swa.update_parameters(net)
            log.append(loss.item())
            it += 1
            if verbose and (it % log_every == 0 or it == n_iters):
                print(f"  iter {it:5d}/{n_iters}  loss {np.mean(log[-log_every:]):.4f}  ({time.time() - t_start:.0f}s)")

    encoder = copy.deepcopy(swa.module).eval()
    return encoder, log
