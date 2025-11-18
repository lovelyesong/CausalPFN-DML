# benchmarks/bonus/07_eval_waae_pfn_only.py
# ----------------------------------------------------------------------
# Evaluate a PFN-lite Two-Head checkpoint that directly predicts
#   (mu0(x), mu1(x)) on the Bonus dataset.
#
# Supported WAAE definitions:
#   - "paper": Jung et al. (r69) definition using global means μ(t) with DR/AIPW
#   - "kpi"  : Project KPI style (group-mean absolute error with group ratios)
#
# Inputs:
#   - --ckpt       : path to two-head PFN checkpoint (must have cfg['two_head']=True)
#   - --bonus_dir  : folder with X.npy / T.npy / Y.npy
#   - --out_json   : where to write the JSON report
#   - --waae_mode  : {"paper","kpi"}
#   - --e_hat      : (optional) path to a precomputed propensity array; if not
#                    provided and mode=paper, the script will fit a simple
#                    out-of-fold logistic regression to estimate e(x).
#
# Outputs:
#   - JSON report with the requested WAAE and diagnostics
#
#
# ----------------------------------------------------------------------

# automatic logistic for propensity score
# --ckpt benchmarks/bonus/outputs_A_twohead/06_pfn_twohead.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_A_twohead_paper.json
# --waae_mode paper

# e_hat for propensity score
# --ckpt benchmarks/bonus/outputs_A_twohead/06_pfn_twohead.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_A_twohead_paper.json
# --waae_mode paper
# --e_hat benchmarks/bonus/outputs/e_hat.npy

# kpi mode
# --ckpt benchmarks/bonus/outputs_A_twohead/06_pfn_twohead.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_A_twohead_kpi.json
# --waae_mode kpi

# paper
# --ckpt benchmarks/bonus/outputs_A_twohead/06_pfn_twohead.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_A_twohead_paper.json
# --waae_mode paper


import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Optional: used only when --waae_mode paper and --e_hat is missing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# ----------------------------- Utilities -----------------------------

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    # fallback: go a few levels up
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]


def load_bonus_arrays(bonus_dir: Path):
    X = np.load(bonus_dir / "X.npy").astype(np.float32)
    T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
    return X, T, Y


def ensure_outdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


# ------------------------- WAAE definitions -------------------------

def waae_kpi(Y, T, mu0_pred, mu1_pred):
    """
    KPI definition (group-mean absolute error with group ratios):
      WAAE = r0 * | mean(mu0 | T=0) - mean(Y | T=0) |
            + r1 * | mean(mu1 | T=1) - mean(Y | T=1) |
    """
    mask0 = (T == 0)
    mask1 = ~mask0
    n = len(T)
    n0 = int(mask0.sum())
    n1 = n - n0
    r0 = n0 / n if n > 0 else 0.0
    r1 = n1 / n if n > 0 else 0.0

    mu0_mean_pred = float(mu0_pred[mask0].mean()) if n0 > 0 else 0.0
    mu1_mean_pred = float(mu1_pred[mask1].mean()) if n1 > 0 else 0.0
    y0_mean = float(Y[mask0].mean()) if n0 > 0 else 0.0
    y1_mean = float(Y[mask1].mean()) if n1 > 0 else 0.0

    err0 = abs(mu0_mean_pred - y0_mean)
    err1 = abs(mu1_mean_pred - y1_mean)
    waae = r0 * err0 + r1 * err1
    return dict(
        waae=float(waae), r0=float(r0), r1=float(r1),
        err0=float(err0), err1=float(err1),
        mu0_mean_pred=float(mu0_mean_pred), mu1_mean_pred=float(mu1_mean_pred),
        y0_mean=float(y0_mean), y1_mean=float(y1_mean)
    )


def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat):
    """
    Paper (r69) definition using global means μ(t) with DR/AIPW:
      WAAE(μ̂) = sum_t | μ̂_bar(t) - μ_bar(t) | * P_N(t)

    Global predicted means:
      μ̂_bar(0) = mean_x μ̂0(x),  μ̂_bar(1) = mean_x μ̂1(x)

    DR/AIPW estimates of true global means:
      μ_bar(0) = 1/N * Σ [ μ̂0(Xi) + 1{Ti=0}/(1-ê(Xi)) * (Yi - μ̂0(Xi)) ]
      μ_bar(1) = 1/N * Σ [ μ̂1(Xi) + 1{Ti=1}/ ê(Xi)    * (Yi - μ̂1(Xi)) ]
    """
    n = len(Y)
    r1 = float((T == 1).mean())
    r0 = 1.0 - r1

    mu0_bar_hat = float(mu0_pred.mean())
    mu1_bar_hat = float(mu1_pred.mean())

    e1 = np.clip(e_hat, 1e-3, 1 - 1e-3)
    e0 = 1.0 - e1
    ind0 = (T == 0).astype(np.float32)
    ind1 = 1.0 - ind0

    aipw0 = mu0_pred + ind0 / e0 * (Y - mu0_pred)
    aipw1 = mu1_pred + ind1 / e1 * (Y - mu1_pred)
    mu0_bar = float(aipw0.mean())
    mu1_bar = float(aipw1.mean())

    err0 = abs(mu0_bar_hat - mu0_bar)
    err1 = abs(mu1_bar_hat - mu1_bar)
    waae = r0 * err0 + r1 * err1

    return dict(
        waae=float(waae), r0=float(r0), r1=float(r1),
        err0=float(err0), err1=float(err1),
        mu0_mean_pred=float(mu0_bar_hat), mu1_mean_pred=float(mu1_bar_hat),
        mu0_mean_true=float(mu0_bar),     mu1_mean_true=float(mu1_bar)
    )


# ------------------------- Propensity helpers -------------------------

def oof_propensity(X, T, n_splits=5, random_state=2025):
    """
    Simple out-of-fold logistic regression to estimate e(x)=P(T=1|X).
    Returns e_hat with the same length as T. Used only if --e_hat is not given.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    e_hat = np.zeros(len(T), dtype=np.float32)
    for tr, te in skf.split(X, T):
        clf = LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")
        clf.fit(X[tr], T[tr])
        e_hat[te] = clf.predict_proba(X[te])[:, 1]
    return np.clip(e_hat, 1e-3, 1 - 1e-3)


# --------------------------- Two-head model ---------------------------

class PFNLiteTwoHead(nn.Module):
    """Two-head MLP for (mu0(x), mu1(x)) prediction."""
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
            d = hidden
        layers += [nn.Linear(d, 2)]  # outputs [mu0, mu1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (n,2)


def load_twohead_mu(X: np.ndarray, ckpt_path: Path):
    """
    Load a two-head PFN checkpoint and return (mu0_pred, mu1_pred).
    Requires ckpt['cfg']['two_head'] == True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    p = ckpt["p"]
    assert p == X.shape[1], f"Input dim mismatch: ckpt p={p}, X has {X.shape[1]}."
    assert bool(cfg.get("two_head", False)) is True, \
        "This script expects a two-head PFN ckpt (cfg['two_head']=True)."

    model = PFNLiteTwoHead(
        in_dim=p,
        hidden=cfg.get("hidden", 256),
        depth=cfg.get("depth", 3),
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        out = model(Xt).cpu().numpy().astype(np.float32)  # (n,2)
        mu0 = out[:, 0]
        mu1 = out[:, 1]
    return mu0, mu1, cfg


# -------------------------------- Main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate PFN two-head (mu0, mu1) on Bonus with WAAE (paper/kpi).")
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus",
                    help="Folder with X.npy / T.npy / Y.npy.")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to two-head PFN checkpoint (e.g., outputs_A_twohead/06_pfn_twohead.ckpt).")
    ap.add_argument("--out_json", type=str, default="benchmarks/bonus/outputs/07_waae_report_A_twohead.json",
                    help="Where to write the JSON report.")
    ap.add_argument("--waae_mode", type=str, default="paper",
                    choices=["paper", "kpi"],
                    help="paper=r69 with DR/AIPW; kpi=group-mean KPI metric.")
    ap.add_argument("--e_hat", type=str, default=None,
                    help="Optional path to a precomputed propensity array. If not provided in paper mode, "
                         "a simple OOF logistic model will be fit automatically.")
    args = ap.parse_args()

    # Resolve paths
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    BONUS_DIR = (repo_root / args.bonus_dir).resolve()
    CKPT = (repo_root / args.ckpt).resolve()
    OUT_JSON = (repo_root / args.out_json).resolve()
    ensure_outdir(OUT_JSON)

    # Load data
    X, T, Y = load_bonus_arrays(BONUS_DIR)

    # Load PFN two-head predictions
    mu0_pfn, mu1_pfn, cfg = load_twohead_mu(X, CKPT)

    # Compute WAAE
    if args.waae_mode == "paper":
        # Propensity: load or fit OOF logistic
        if args.e_hat is not None:
            e_path = (repo_root / args.e_hat).resolve()
            e_hat = np.load(e_path).astype(np.float32)
            print(f"[info] loaded e_hat from {e_path} (shape={e_hat.shape})")
        else:
            print("[info] --e_hat not given; fitting OOF logistic propensity (5-fold)…")
            e_hat = oof_propensity(X, T, n_splits=5, random_state=2025)
        stats = waae_paper(Y, T, mu0_pfn, mu1_pfn, e_hat)
    else:
        stats = waae_kpi(Y, T, mu0_pfn, mu1_pfn)

    # Assemble report
    report = {
        "mode": args.waae_mode,
        "PFN_twohead": stats,
        "ckpt_evaluated": str(CKPT),
        "notes": (
            "PFN two-head outputs (mu0, mu1) were used directly. "
            "In 'paper' mode, DR/AIPW uses these heads as outcome models; "
            "propensities are from --e_hat if provided, else OOF logistic."
        ),
        "cfg": cfg,
    }

    print("[WAAE Report]")
    print(json.dumps(report, indent=2))
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
