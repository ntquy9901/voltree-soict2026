from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[1]
if str(REPO / "baselines" / "common") not in sys.path:
    sys.path.insert(0, str(REPO / "baselines" / "common"))
import feature_panel as D

FL = D.FL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOPK = 10

# ----------------------------------------------------------------- correlation top-k graph (train-only)
def _std_cols(df):
    z = (df - df.mean()) / df.std().replace(0, np.nan)
    return z.fillna(0.0).to_numpy(float), z.notna().to_numpy(float)

def build_graph(train, tickers):
    piv = train.pivot_table(index="date", columns="ticker", values="logpk").reindex(columns=tickers)
    X, Xm = _std_cols(piv)
    corr = (X.T @ X) / np.maximum(Xm.T @ Xm, 1.0)
    np.fill_diagonal(corr, -np.inf)
    n = len(tickers)
    W = np.zeros((n, n))
    for i in range(n):
        top = np.argsort(corr[i])[::-1][:TOPK]
        W[i, top] = np.clip(corr[i, top], 0.0, None)
    W /= np.maximum(np.abs(W).sum(1, keepdims=True), 1e-12)
    return W

# ----------------------------------------------------------------- faithful GNNHAR (arXiv:2308.01419)
class GraphConvLayer(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_f, out_f))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        self.bias = nn.Parameter(torch.ones(1, out_f)) if bias else None

    def forward(self, x, adj):
        h = torch.matmul(x, self.weight)
        out = torch.matmul(adj, h)
        return out + self.bias if self.bias is not None else out

class GNNHAR(nn.Module):
    def __init__(self, in_f, n_hid, n_gcn):
        super().__init__()
        self.linear1 = nn.Linear(in_f, 1, bias=True)
        nn.init.constant_(self.linear1.bias, 1.0)
        gcns = [GraphConvLayer(in_f, n_hid, bias=False)]
        gcns += [GraphConvLayer(n_hid, n_hid, bias=False) for _ in range(n_gcn - 1)]
        self.gcns = nn.ModuleList(gcns)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        h1 = self.linear1(x)
        hg = x
        for gcn in self.gcns:
            hg = self.relu(gcn(hg, adj))
        hg = self.mlp1(hg)
        return self.relu(h1 + hg).squeeze(-1)

def _qlike_loss(pred, y, mask):
    f = pred + 1e-4
    tf = torch.clamp(y, min=FL) / f
    loss = tf - torch.log(tf)
    return (loss * mask).sum() / mask.sum()

# ----------------------------------------------------------------- fold tensors + gather plumbing
def build_fold_tensors(fold, tickers, feats, tr_dates_mask_date):
    dts = np.sort(fold["date"].unique())
    dpos = {d: i for i, d in enumerate(dts)}
    cpos = {t: j for j, t in enumerate(tickers)}
    Dn, N = len(dts), len(tickers)
    r = fold["date"].map(dpos).to_numpy()
    c = fold["ticker"].map(cpos).to_numpy()
    mask = np.zeros((Dn, N), np.float32)
    mask[r, c] = 1.0
    Y = np.zeros((Dn, N), np.float32)
    Y[r, c] = fold["y"].to_numpy(np.float32)
    is_train_date = np.array([tr_dates_mask_date(d) for d in dts])
    train_cells = mask.astype(bool) & is_train_date[:, None]
    X = np.zeros((Dn, N, len(feats)), np.float32)
    for fi, col in enumerate(feats):
        mm = np.zeros((Dn, N), np.float32)
        mm[r, c] = fold[col].to_numpy(np.float32)
        vals = mm[train_cells]
        mu, sd = float(vals.mean()), float(vals.std())
        sd = sd if sd > 1e-12 else 1.0
        z = (mm - mu) / sd
        z[~mask.astype(bool)] = 0.0
        X[:, :, fi] = z
    sc = 1.0 / max(float(Y[train_cells].mean()), FL)
    return X, Y, Y * sc, mask, sc, dpos, cpos, dts

def _row_in_te(tef_dates, dpos, te_idx):
    te_pos = {int(p): i for i, p in enumerate(te_idx)}
    return np.array([te_pos[int(p)] for p in tef_dates.map(dpos)])

def row_preds(model, X, adj, date_idx, row_in, cidx, sc):
    model.eval()
    with torch.no_grad():
        pm = model(X[date_idx], adj).cpu().numpy()
    return np.maximum(pm[row_in, cidx] / sc, FL)
