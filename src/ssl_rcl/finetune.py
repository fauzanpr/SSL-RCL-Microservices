"""Fine-tuning lokalisasi root-cause service di atas encoder hasil pre-training.

Alur per kasus:
  x [N, T, K] + mask [N, K]
    -> input encoder per service: concat(x, mask diulang sepanjang T) -> [N, T, 2K]
    -> encoder -> representasi per timestamp [N, T, D]
    -> fitur service: concat(maxpool_pasca, maxpool_pasca - maxpool_pra) -> [N, 2D]
    -> head MLP -> skor per service [N] -> softmax di antara service kasus tersebut
Loss: cross-entropy terhadap indeks root service. Jumlah service N boleh berbeda antar
sistem, karena encoder dan head dipakai bersama untuk setiap service.

Mode:
- FineLast: encoder dibekukan, hanya head yang dilatih.
- FineAll : encoder + head dilatih (lr encoder lebih kecil).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import collate_cases


def encoder_input(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """x [..., T, K], channel_mask [..., K] -> [..., T, 2K]."""
    m = channel_mask.unsqueeze(-2).expand_as(x)
    return torch.cat([x, m], dim=-1)


def service_features(encoder, x, channel_mask, inject_offset: int, chunk: int = 256) -> torch.Tensor:
    """x [S, T, K] -> fitur [S, 2D]. Dipotong per `chunk` service agar hemat memori."""
    feats = []
    for i in range(0, x.size(0), chunk):
        h = encoder(encoder_input(x[i:i + chunk], channel_mask[i:i + chunk]), mask="all_true")  # [s, T, D]
        h_pre = h[:, :inject_offset].amax(dim=1)
        h_post = h[:, inject_offset:].amax(dim=1)
        feats.append(torch.cat([h_post, h_post - h_pre], dim=-1))
    return torch.cat(feats)


class ScoreHead(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


def _scatter_scores(flat_scores, service_mask):
    scores = torch.full(service_mask.shape, float("-inf"), device=flat_scores.device)
    scores[service_mask] = flat_scores
    return scores


def ranks_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Rank (1 = terbaik) root service = 1 + jumlah service dengan skor lebih tinggi (sama seperti heuristik)."""
    true = scores.gather(1, labels[:, None])
    return ((scores > true).sum(1) + 1).cpu().numpy()


def summarize_ranks(r) -> dict:
    r = np.asarray(r)
    out = {f"AC@{k}": float((r <= k).mean()) for k in (1, 3, 5)}
    out["Avg@5"] = float(np.mean([(r <= k).mean() for k in range(1, 6)]))
    out["n"] = int(len(r))
    return out


# ---------------------------------------------------------------- helper batch
def _batches(samples, batch_size, shuffle, rng):
    idx = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
    for b in range(0, len(idx), batch_size):
        yield collate_cases([samples[i] for i in idx[b:b + batch_size]])


def _check_offset(samples):
    offs = {s["inject_offset"] for s in samples}
    if len(offs) != 1:
        raise ValueError(f"inject_offset harus sama untuk semua kasus, dapat {offs}")
    return offs.pop()


# ---------------------------------------------------------------- FineLast
@torch.no_grad()
def precompute_features(encoder, samples, device, batch_size=8):
    """Untuk FineLast: fitur service sekali hitung dengan encoder beku."""
    encoder.eval()
    off = _check_offset(samples)
    out = []
    for s in samples:
        x = torch.from_numpy(s["x"]).to(device)
        m = torch.from_numpy(s["mask"]).to(device)
        out.append(service_features(encoder, x, m, off).cpu())
    return out


def _feature_batches(feats, samples, idx):
    n_max = max(feats[i].size(0) for i in idx)
    D = feats[idx[0]].size(1)
    f = torch.zeros(len(idx), n_max, D)
    sm = torch.zeros(len(idx), n_max, dtype=torch.bool)
    for j, i in enumerate(idx):
        n = feats[i].size(0)
        f[j, :n] = feats[i]
        sm[j, :n] = True
    y = torch.tensor([samples[i]["label"] for i in idx])
    return f, sm, y


def _eval_head_on_feats(head, feats, samples, device, batch_size=64):
    head.eval()
    ranks, losses = [], []
    with torch.no_grad():
        for b in range(0, len(samples), batch_size):
            idx = list(range(b, min(b + batch_size, len(samples))))
            f, sm, y = _feature_batches(feats, samples, idx)
            f, sm, y = f.to(device), sm.to(device), y.to(device)
            scores = _scatter_scores(head(f[sm]), sm)
            losses.append(F.cross_entropy(scores, y, reduction="sum").item())
            ranks.append(ranks_from_scores(scores, y))
    return float(np.sum(losses) / len(samples)), np.concatenate(ranks)


def train_finelast(encoder, train, val, device, epochs=200, batch_size=16, lr=1e-3, weight_decay=1e-4,
                   patience=30, seed=42, verbose=False):
    """Latih head saja. Model terbaik dipilih berdasarkan loss validasi."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    f_tr, f_va = precompute_features(encoder, train, device), precompute_features(encoder, val, device)
    head = ScoreHead(f_tr[0].size(1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, wait, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        head.train()
        perm = rng.permutation(len(train))
        for b in range(0, len(perm), batch_size):
            f, sm, y = _feature_batches(f_tr, train, perm[b:b + batch_size])
            f, sm, y = f.to(device), sm.to(device), y.to(device)
            loss = F.cross_entropy(_scatter_scores(head(f[sm]), sm), y)
            opt.zero_grad(); loss.backward(); opt.step()
        va_loss, va_r = _eval_head_on_feats(head, f_va, val, device)
        hist.append(dict(epoch=ep, val_loss=va_loss, **summarize_ranks(va_r)))
        if va_loss < best - 1e-4:
            best, best_state, wait = va_loss, copy.deepcopy(head.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    head.load_state_dict(best_state)
    if verbose:
        print(f"  FineLast: berhenti di epoch {len(hist)}, best val loss {best:.4f}")
    return dict(encoder=encoder, head=head, history=pd.DataFrame(hist))


# ---------------------------------------------------------------- FineAll
def _forward_cases(encoder, head, batch, device, inject_offset):
    x = batch["x"].to(device)
    cm = batch["channel_mask"].to(device)
    sm = batch["service_mask"].to(device)
    feats = service_features(encoder, x[sm], cm[sm], inject_offset)
    return _scatter_scores(head(feats), sm)


def evaluate_cases(encoder, head, samples, device, batch_size=4):
    encoder.eval(); head.eval()
    off = _check_offset(samples)
    ranks, losses = [], []
    with torch.no_grad():
        for batch in _batches(samples, batch_size, False, None):
            scores = _forward_cases(encoder, head, batch, device, off)
            y = batch["label"].to(device)
            losses.append(F.cross_entropy(scores, y, reduction="sum").item())
            ranks.append(ranks_from_scores(scores, y))
    return float(np.sum(losses) / len(samples)), np.concatenate(ranks)


def train_fineall(encoder, train, val, device, epochs=30, batch_size=4, lr_head=1e-3, lr_encoder=1e-4,
                  weight_decay=1e-4, patience=8, seed=42, verbose=False, init_head=None, micro_batch=1):
    """Latih encoder + head. Model terbaik dipilih berdasarkan loss validasi.

    batch_size = batch efektif (kasus per langkah optimizer). Agar muat di memori kecil
    (mis. 8 GB), forward/backward dijalankan per `micro_batch` kasus dengan gradient
    accumulation. Hasilnya setara dengan batch penuh, karena loss dirata-rata per kasus.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    off = _check_offset(train)
    encoder = copy.deepcopy(encoder).to(device)
    D = encoder.output_dims
    head = (copy.deepcopy(init_head) if init_head is not None else ScoreHead(2 * D)).to(device)
    opt = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": lr_encoder},
        {"params": head.parameters(), "lr": lr_head},
    ], weight_decay=weight_decay)
    best, best_state, wait, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        encoder.train(); head.train()
        perm = rng.permutation(len(train))
        for b in range(0, len(perm), batch_size):
            chunk = perm[b:b + batch_size]
            opt.zero_grad()
            for mb in range(0, len(chunk), micro_batch):
                batch = collate_cases([train[i] for i in chunk[mb:mb + micro_batch]])
                scores = _forward_cases(encoder, head, batch, device, off)
                loss = F.cross_entropy(scores, batch["label"].to(device), reduction="sum") / len(chunk)
                loss.backward()
            opt.step()
        if device.type == "mps":
            torch.mps.empty_cache()
        va_loss, va_r = evaluate_cases(encoder, head, val, device, batch_size=micro_batch)
        hist.append(dict(epoch=ep, val_loss=va_loss, **summarize_ranks(va_r)))
        if verbose:
            print(f"  FineAll epoch {ep:2d} val_loss {va_loss:.4f} val AC@1 {hist[-1]['AC@1']:.3f} ({time.time() - t0:.0f}s)")
        if va_loss < best - 1e-4:
            best, wait = va_loss, 0
            best_state = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(head.state_dict()))
        else:
            wait += 1
            if wait >= patience:
                break
    encoder.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    return dict(encoder=encoder, head=head, history=pd.DataFrame(hist))


def predict_table(encoder, head, samples, device) -> pd.DataFrame:
    """Rank root service per kasus (untuk analisis per sistem / per fault)."""
    _, r = evaluate_cases(encoder, head, samples, device)
    return pd.DataFrame(dict(name=[s["name"] for s in samples], system=[s["system"] for s in samples],
                             fault=[s["fault"] for s in samples], rank=r))
