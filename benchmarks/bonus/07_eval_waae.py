# benchmarks/bonus/07_eval_waae.py
# ---------------------------------------------------------------
# Evaluate WAAE for:
#   (1) DML baseline using mu0_hat.npy, mu1_hat.npy (from step 04)
#   (2) PFN(+DML) combo using tau_pred from a PFN-lite checkpoint:
#       mu0_pfn = mu0_hat_dml
#       mu1_pfn = mu0_hat_dml + tau_pfn
#
# CLI options:
#   --ckpt <path/to/06_pfnlite.ckpt>     # which PFN checkpoint to evaluate
#   --out_json <path/to/output.json>     # where to store the report
#
# Outputs:
#   - JSON report with DML WAAE, PFN+DML WAAE, relative improvement,
#     and whether the target (>=10% reduction vs DML) is met.
# # ---------------------------------------------------------------

# (B)로 한거 PairNet 으로 WAAE한거
# --ckpt benchmarks/bonus/outputs_B/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_B.json

# (c)로 한거
# --ckpt benchmarks/bonus/outputs_C/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_C.json

# (B)로 한거 Jung et al. (2020)기준 WAAE
# --ckpt benchmarks/bonus/outputs_B/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_B_paper.json
# --waae_mode paper

# (c)로 한거  #30을 21, 24로 바꿔보기 -> lambda 30에 gamma 8이 베스트!!!!!!!!!!!!!!!!
# --ckpt benchmarks/bonus/outputs_Cx30/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_Cx30_paper.json
# --waae_mode paper

# (B)로 한거  #30을 21, 24로 바꿔보기 -> lambda 30에 gamma 8이 베스트!!!! 와 비교하는 B
# --ckpt benchmarks/bonus/outputs_Bx1p44/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_Bx1p44_paper.json
# --waae_mode paper

# (a) PFN
# --ckpt benchmarks/bonus/outputs_A/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_A_paper.json
# --waae_mode paper


import os
import json
import argparse
import numpy as np
from pathlib import Path
import torch


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]


def load_bonus_arrays(bonus_dir: Path):
    X = np.load(bonus_dir / "X.npy").astype(np.float32)
    T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
    return X, T, Y

def waae_kpi(Y, T, mu0_pred, mu1_pred): #작년 pairnet방법
    """
    KPI definition used in your project:
      WAAE = r0 * | mean(mu0 | T=0) - mean(Y | T=0) |
            + r1 * | mean(mu1 | T=1) - mean(Y | T=1) |
    """
    mask0 = (T == 0)
    mask1 = (T == 1)
    n = len(T)
    n0 = int(mask0.sum())
    n1 = int(mask1.sum())
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
        mu0_mean_pred=float(mu0_mean_pred),
        mu1_mean_pred=float(mu1_mean_pred),
        y0_mean=float(y0_mean), y1_mean=float(y1_mean)
    )


def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_hat_dml, mu1_hat_dml): # Jung et al. 2020 방법
    """
    Paper (r69) definition:
      WAAE(μ̂) = sum_t | μ̂_bar(t) - μ_bar(t) | * P_N(t)
      where μ̂_bar(t) is the global (unconditional) mean predicted under t,
      μ_bar(t) is the true global mean under t, estimated by DR/AIPW.

    DR/AIPW estimators:
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

    aipw0 = mu0_hat_dml + ind0 / e0 * (Y - mu0_hat_dml)
    aipw1 = mu1_hat_dml + ind1 / e1 * (Y - mu1_hat_dml)
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


# def waae_from_predictions(Y, T, mu0_pred, mu1_pred):
#     """
#     Weighted Absolute Arm Error (WAAE):
#       r0 * | mean(mu0_pred[T==0]) - mean(Y[T==0]) |
#     + r1 * | mean(mu1_pred[T==1]) - mean(Y[T==1]) |
#     where r_t are group ratios n_t / n.
#     """
#     mask0 = (T == 0)
#     mask1 = (T == 1)
#     n = len(T)
#     n0 = int(mask0.sum())
#     n1 = int(mask1.sum())
#     r0 = n0 / n if n > 0 else 0.0
#     r1 = n1 / n if n > 0 else 0.0
#
#     mu0_mean_pred = float(mu0_pred[mask0].mean()) if n0 > 0 else 0.0
#     mu1_mean_pred = float(mu1_pred[mask1].mean()) if n1 > 0 else 0.0
#     y0_mean = float(Y[mask0].mean()) if n0 > 0 else 0.0
#     y1_mean = float(Y[mask1].mean()) if n1 > 0 else 0.0
#
#     err0 = abs(mu0_mean_pred - y0_mean)
#     err1 = abs(mu1_mean_pred - y1_mean)
#     waae = r0 * err0 + r1 * err1
#     return dict(
#         waae=float(waae), r0=float(r0), r1=float(r1),
#         err0=float(err0), err1=float(err1),
#         mu0_mean_pred=float(mu0_mean_pred),
#         mu1_mean_pred=float(mu1_mean_pred),
#         y0_mean=float(y0_mean), y1_mean=float(y1_mean)
#     )


# Must match 06_train_pfn.py architecture
class PFNLite(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [torch.nn.Dropout(p=dropout)]
            d = hidden
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (n,1)


def load_tau_pred_with_ckpt(X: np.ndarray, ckpt_path: Path) -> np.ndarray:
    """Load PFN-lite checkpoint and compute tau_pred(X)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    p = ckpt["p"]
    assert p == X.shape[1], f"Input dim mismatch: ckpt p={p}, X has {X.shape[1]}."

    model = PFNLite(in_dim=p, hidden=cfg["hidden"], depth=cfg["depth"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        tau = model(Xt).squeeze(1).cpu().numpy().astype(np.float32)
    return tau


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate WAAE (KPI or Paper/r69) for DML baseline and PFN(+DML) combo.")
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus",
                    help="Folder with X.npy/T.npy/Y.npy.")
    ap.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs",
                    help="Folder where mu0_hat.npy, mu1_hat.npy, e_hat.npy live; JSON report is stored here by default.")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to 06_pfnlite.ckpt to evaluate (e.g., outputs_B/06_pfnlite.ckpt).")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Path to save JSON report (default: outputs/07_waae_report.json).")
    ap.add_argument("--waae_mode", type=str, default="kpi",
                    choices=["kpi", "paper"],
                    help="kpi: project KPI definition; paper: r69 WAAE with DR/AIPW.")
    args = ap.parse_args()

    # Resolve paths
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    BONUS_DIR = (repo_root / args.bonus_dir).resolve()
    OUT_DIR = (repo_root / args.outputs_dir).resolve()
    CKPT = (repo_root / args.ckpt).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = (repo_root / args.out_json).resolve() if args.out_json else (OUT_DIR / "07_waae_report.json")

    # Load arrays
    X, T, Y = load_bonus_arrays(BONUS_DIR)

    # Load DML artifacts
    mu0_dml = np.load(OUT_DIR / "mu0_hat.npy")
    mu1_dml = np.load(OUT_DIR / "mu1_hat.npy")
    if args.waae_mode == "paper":
        e_path = OUT_DIR / "e_hat.npy"
        if not e_path.exists():
            raise FileNotFoundError(f"Missing {e_path}. Save e_hat.npy in step 04 or switch to --waae_mode kpi.")
        e_hat = np.load(e_path)
    else:
        e_hat = None

    # PFN(+DML) combo
    tau_pfn = load_tau_pred_with_ckpt(X, CKPT)
    mu0_pfn = mu0_dml
    mu1_pfn = mu0_dml + tau_pfn
    mu1_pfn_raw = mu0_dml + tau_pfn  # <-- 먼저 이걸 만든 다음

    ### A (PFN) 할떄 아래 블럭 닫고하기
    # Try to load zeta weights; if missing, fall back to raw PFN
    zeta_path = OUT_DIR / "05_zeta_bonus.npy"
    gamma = 0.8  # ↓ try 0.3 ~ 0.6; smaller = PFN 영향 축소 → DML 쪽으로 수축
    if zeta_path.exists():
        zeta = np.load(str(zeta_path))
        w = zeta / (zeta.mean() + 1e-8)
        w = gamma * w  # mean(w) ≈ gamma (not 1) → DML 쪽으로 전체 수축
        mu1_pfn = (1.0 - w) * mu1_dml + w * mu1_pfn_raw
    else:
        print(f"[warn] {zeta_path} not found; using raw PFN tau (no blending).")
        mu1_pfn = mu1_pfn_raw
    ### A 할때 위 블럭 닫고하기

    # Compute stats
    if args.waae_mode == "paper":
        dml_stats = waae_paper(Y, T, mu0_dml, mu1_dml, e_hat, mu0_dml, mu1_dml)
        pfn_stats = waae_paper(Y, T, mu0_pfn, mu1_pfn, e_hat, mu0_dml, mu1_dml)
    else:
        dml_stats = waae_kpi(Y, T, mu0_dml, mu1_dml)
        pfn_stats = waae_kpi(Y, T, mu0_pfn, mu1_pfn)

    dml_waae = dml_stats["waae"]
    pfn_waae = pfn_stats["waae"]
    rel_impr = (dml_waae - pfn_waae) / dml_waae if dml_waae > 0 else 0.0
    target_ok = rel_impr >= 0.10

    report = {
        "mode": args.waae_mode,
        "DML": dml_stats,
        "PFN_plus_DML": pfn_stats,
        "relative_improvement": float(rel_impr),
        "target_met_10pct_reduction": bool(target_ok),
        "ckpt_evaluated": str(CKPT),
        "notes": "Paper mode uses DR/AIPW to estimate global means μ(t). "
                 "PFN(+DML) uses mu0 from DML and mu1 = mu0 + tau_pfn."
    }

    print("[WAAE Report]")
    print(json.dumps(report, indent=2))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] saved -> {out_json}")


if __name__ == "__main__":
    main()
