"""Baseline heuristik: ranking service berdasarkan rata-rata |z| pasca-injeksi atas kanal aktif."""

import numpy as np
import pandas as pd


def heuristic_scores(s: dict) -> np.ndarray:
    post = np.abs(s["x"][:, s["inject_offset"]:])          # [N, T_post, K]
    m = s["mask"][:, None, :]
    return (post * m).sum((1, 2)) / np.maximum(m.sum((1, 2)) * post.shape[1], 1)


def heuristic_rank(s: dict) -> int:
    sc = heuristic_scores(s)
    return int((sc > sc[s["label"]]).sum()) + 1


def heuristic_table(samples) -> pd.DataFrame:
    return pd.DataFrame(dict(name=[s["name"] for s in samples], system=[s["system"] for s in samples],
                             fault=[s["fault"] for s in samples], rank=[heuristic_rank(s) for s in samples]))
