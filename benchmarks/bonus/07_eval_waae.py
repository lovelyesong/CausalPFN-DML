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

# (B) PairNet KPI
# --ckpt benchmarks/bonus/outputs_B/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_B.json

# (C) KPI
# --ckpt benchmarks/bonus/outputs_C/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_C.json

# (B) Jung et al. (2020) WAAE (paper mode)
# --ckpt benchmarks/bonus/outputs_B/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_B_paper.json
# --waae_mode paper

# (C) paper mode
# --ckpt benchmarks/bonus/outputs_Cx30/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_Cx30_paper.json
# --waae_mode paper

# (A) PFN
# --ckpt benchmarks/bonus/outputs_A/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_A_paper.json
# --waae_mode paper

# official CausalPFN (06_pfn_cate_blend.py output)
# --ckpt benchmarks/bonus/outputs_CPFN_official/06_pfnlite.ckpt
# --out_json benchmarks/bonus/outputs/07_waae_report_CPFN_paper.json
# --waae_mode paper

# python benchmarks/bonus/07_eval_waae.py \
#   --ckpt benchmarks/bonus/outputs_CPFN_official_noshift/06_pfnlite.ckpt \
#   --outputs_dir benchmarks/bonus/outputs \
#   --out_json benchmarks/bonus/outputs/07_waae_report_CPFN_noshift_paper.json \
#   --waae_mode paper

# benchmarks/bonus/07_eval_waae.py
# ---------------------------------------------------------------
# Evaluate WAAE for:
#   (1) DML baseline using mu0_hat.npy, mu1_hat.npy (from step 04)
#   (2) PFN(+DML) combo using tau_pred from a PFN-lite checkpoint
#       with selectable blending:
#         - alpha: mean-preserving additive (auto alpha★)
#         - gamma: convex blend with w = gamma * zeta / E[zeta]
#         - none : raw mu1 = mu0_dml + tau_pfn
#
# Example:
#   python benchmarks/bonus/07_eval_waae.py \
#     --ckpt benchmarks/bonus/outputs_CPFN_official/06_pfnlite.ckpt \
#     --outputs_dir benchmarks/bonus/outputs \
#     --out_json benchmarks/bonus/outputs/07_waae_report_CPFN_paper.json \
#     --waae_mode paper --blend alpha --dump_preds
# ---------------------------------------------------------------


# no shift (no zeta)
# python benchmarks/bonus/07_eval_waae.py \
#   --ckpt benchmarks/bonus/outputs_CPFN_noshift/06_pfnlite.ckpt \
#   --outputs_dir benchmarks/bonus/outputs \
#   --shift_outputs_dir benchmarks/bonus/outputs_noshift \
#   --out_json benchmarks/bonus/outputs/07_waae_report_noshift_paper_gamma.json \
#   --waae_mode paper \
#   --blend gamma --gamma 0.8 --zeta_norm none --dump_preds

# shift yes (zeta)
# python benchmarks/bonus/07_eval_waae.py \
#   --ckpt benchmarks/bonus/outputs_CPFN_shift/06_pfnlite.ckpt \
#   --outputs_dir benchmarks/bonus/outputs \
#   --shift_outputs_dir benchmarks/bonus/outputs \
#   --out_json benchmarks/bonus/outputs/07_waae_report_shift_paper_gamma.json \
#   --waae_mode paper \
#   --blend gamma --gamma 0.8 --zeta_norm none --dump_preds


# --ckpt benchmarks/bonus/outputs_CPFN_shift/06_pfnlite.ckpt
# --outputs_dir benchmarks/bonus/outputs
# --shift_outputs_dir benchmarks/bonus/outputs
# --out_json benchmarks/bonus/outputs/07_waae_shift_paper_alpha.json
# --waae_mode paper
# --blend alpha
# --dump_preds

#
# --ckpt benchmarks/bonus/outputs_A_twohead/06_pfn_twohead.ckpt
# --outputs_dir benchmarks/bonus/outputs
# --out_json benchmarks/bonus/outputs/07_waae_pfnonly_paper.json
# --waae_mode paper
# --use_ckpt_mu

# benchmarks/bonus/07_eval_waae.py
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


def waae_kpi(Y, T, mu0_pred, mu1_pred):
    """
    Project KPI:
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


def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_hat_dml, mu1_hat_dml):
    """
    Paper (r69) definition:
      WAAE(μ̂) = sum_t | μ̂_bar(t) - μ_bar(t) | * P_N(t)
    DR/AIPW:
      μ_bar(0) = 1/N * Σ [ μ̂0(Xi) + 1{Ti=0}/(1-ê(Xi)) * (Yi - μ̂0(Xi)) ]
      μ_bar(1) = 1/N * Σ [ μ̂1(Xi) + 1{Ti=1}/ ê(Xi)    * (Yi - μ̂1(Xi)) ]
    """
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


def ate_dr(Y, T, e_hat, mu0_hat, mu1_hat):
    """
    DR/AIPW ATE:
      τ_DR = 1/N Σ [ (μ1_hat - μ0_hat)
                     + 1{T=1}/e(X)*(Y-μ1_hat)
                     - 1{T=0}/(1-e(X))*(Y-μ0_hat) ]
    """
    n = len(Y)
    e1 = np.clip(e_hat, 1e-3, 1 - 1e-3)
    e0 = 1.0 - e1
    ind1 = (T == 1).astype(np.float32)
    ind0 = 1.0 - ind1
    term = (mu1_hat - mu0_hat) + ind1 / e1 * (Y - mu1_hat) - ind0 / e0 * (Y - mu0_hat)
    return float(term.mean())


# (Legacy only) PFN-lite — used only if old checkpoints must be rebuilt
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


def load_tau_pred_with_ckpt(X, ckpt_path, device="cpu"):
    """
    Priority:
      1) npz with 'tau_pfn'
      2) torch ckpt with 'tau_pfn'
      3) (legacy) rebuild PFNLite from cfg/state_dict
    """
    n = len(X)

    # 1) npz
    try:
        with np.load(ckpt_path) as z:
            if "tau_pfn" in z:
                tau = z["tau_pfn"].astype(np.float32).reshape(-1)
                assert len(tau) == n
                return tau
    except Exception:
        pass

    # 2) torch ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "tau_pfn" in ckpt:
        tau = ckpt["tau_pfn"]
        if torch.is_tensor(tau):
            tau = tau.detach().cpu().numpy()
        tau = np.asarray(tau, dtype=np.float32).reshape(-1)
        assert len(tau) == n
        return tau

    # 3) legacy rebuild
    need_keys = {"cfg", "p", "state_dict"}
    if isinstance(ckpt, dict) and need_keys.issubset(set(ckpt.keys())):
        cfg = ckpt["cfg"]
        p = int(ckpt["p"])
        sd = ckpt["state_dict"]
        for k in ("hidden", "depth", "dropout"):
            if k not in cfg:
                raise KeyError(f"legacy path: cfg['{k}'] missing")

        model = PFNLite(in_dim=p,
                        hidden=cfg["hidden"],
                        depth=cfg["depth"],
                        dropout=cfg["dropout"]).to(device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X.astype(np.float32)).to(device)
            tau = model(Xt).detach().cpu().numpy().astype(np.float32).reshape(-1)
        assert len(tau) == n
        return tau

    raise KeyError(
        "No 'tau_pfn' found and legacy keys (cfg/p/state_dict) not present. "
        "Please evaluate with a ckpt that contains 'tau_pfn' or keep the legacy reconstruction path."
    )

class PFNLiteTwoHead(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [torch.nn.Dropout(p=dropout)]
            d = hidden
        layers += [torch.nn.Linear(d, 2)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):  # returns (n,2): [:,0]=mu0, [:,1]=mu1
        return self.net(x)

def load_mu_from_twohead_ckpt(X, ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    # 1) ckpt에 이미 저장돼 있으면 바로 사용
    if isinstance(ckpt, dict) and ("mu0_pfn" in ckpt) and ("mu1_pfn" in ckpt):
        mu0 = ckpt["mu0_pfn"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        mu1 = ckpt["mu1_pfn"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        assert len(mu0) == len(X) == len(mu1)
        return mu0, mu1
    # 2) 두헤드 모델로부터 추론
    if isinstance(ckpt, dict) and ("cfg" in ckpt) and ckpt["cfg"].get("two_head", False) and ("model" in ckpt) and ("p" in ckpt):
        p = int(ckpt["p"]); cfg = ckpt["cfg"]
        hidden = int(cfg.get("hidden", 256)); depth = int(cfg.get("depth", 3)); dropout = float(cfg.get("dropout", 0.0))
        model = PFNLiteTwoHead(in_dim=p, hidden=hidden, depth=depth, dropout=dropout).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X.astype(np.float32)).to(device)
            out = model(Xt).detach().cpu().numpy().astype(np.float32)
        mu0, mu1 = out[:,0], out[:,1]
        assert len(mu0) == len(X) == len(mu1)
        return mu0, mu1
    raise KeyError("Two-head μ0/μ1 not found: ckpt must contain mu0_pfn/mu1_pfn OR (cfg.two_head & model & p).")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate WAAE (KPI or Paper/r69) or τ-ATE for PFN-only."
    )
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus",
                    help="Folder with X.npy/T.npy/Y.npy.")
    ap.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs",
                    help="Folder where mu0_hat.npy, mu1_hat.npy, e_hat.npy live; JSON report is stored here by default.")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to 06_pfnlite.ckpt to evaluate (e.g., outputs_B/06_pfnlite.ckpt).")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Path to save JSON report (default: outputs/07_waae_report.json).")
    ap.add_argument("--waae_mode", type=str, default="kpi",
                    choices=["kpi", "paper", "tau"],
                    help="kpi/paper: need μ0/μ1; tau: PFN-only ATE via DR vs mean(tau_pfn).")
    ap.add_argument("--blend", type=str, default="alpha",
                    choices=["alpha", "gamma", "none"],
                    help="alpha: mean-preserving; gamma: convex blend; none: raw mu1 = mu0_dml + tau_pfn. (ignored for --waae_mode tau)")
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="Gamma for 'gamma' blend (w = gamma * zeta / E[zeta]). Ignored for tau mode.")
    ap.add_argument("--dump_preds", action="store_true",
                    help="If set, dump mu1 predictions and auxiliary metrics for debugging.")
    ap.add_argument("--shift_outputs_dir", type=str, default=None,
                    help="Folder to load 05_zeta_bonus.npy from. If None, use outputs_dir. "
                         "If file is missing, no-shift blending is used.")
    ap.add_argument("--zeta_norm", type=str, default="mean", choices=["mean", "none", "p95"],
                    help="Gamma blend: "
                         "'mean' uses w=gamma*zeta/mean(zeta), "
                         "'none' uses w=gamma*zeta, "
                         "'p95' → w=gamma*zeta/quantile(zeta,0.95).")
    # ... 기존 argparse 정의에 이어서
    ap.add_argument("--use_ckpt_mu", action="store_true",
                    help="Use PFN two-head mu0/mu1 from ckpt (or infer via model) and evaluate PFN-only WAAE. Blending is ignored.")

    args = ap.parse_args()

    # Resolve paths
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    BONUS_DIR = (repo_root / args.bonus_dir).resolve()
    OUT_DIR = (repo_root / args.outputs_dir).resolve()
    CKPT = (repo_root / args.ckpt).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = (repo_root / args.out_json).resolve() if args.out_json else (OUT_DIR / "07_waae_report.json")
    SHIFT_DIR = (repo_root / (args.shift_outputs_dir or args.outputs_dir)).resolve()

    # Load arrays
    X, T, Y = load_bonus_arrays(BONUS_DIR)

    # Load PFN tau
    tau_pfn = load_tau_pred_with_ckpt(X, CKPT)
    print("[debug] tau_pfn mean/std:", float(np.mean(tau_pfn)), float(np.std(tau_pfn)))

    # ----- τ 모드: PFN-only 평가 (WAAE가 아니라 ATE 오차) -----
    if args.waae_mode == "tau":
        # DR ATE 계산을 위해 DML의 μ0/μ1/e_hat 필요 (파일은 OUT_DIR에 존재)
        e_path = OUT_DIR / "e_hat.npy"
        if not e_path.exists():
            raise FileNotFoundError(f"[tau] Missing {e_path}. Need e_hat.npy (and mu0/mu1) to compute DR ATE.")
        e_hat = np.load(e_path)
        mu0_dml = np.load(OUT_DIR / "mu0_hat.npy")
        mu1_dml = np.load(OUT_DIR / "mu1_hat.npy")

        ate_dr_val = ate_dr(Y, T, e_hat, mu0_dml, mu1_dml)
        ate_pfn = float(np.mean(tau_pfn))
        abs_err = float(abs(ate_pfn - ate_dr_val))

        report = {
            "mode": "tau",
            "tau_eval": {
                "ATE_DR": ate_dr_val,
                "ATE_PFN_mean": ate_pfn,
                "abs_error": abs_err
            },
            "ckpt_evaluated": str(CKPT),
            "notes": "τ-mode compares mean(tau_pfn) vs DR ATE; blending options are ignored."
        }
        if args.blend != "alpha":
            print(f"[info] --waae_mode tau: ignoring --blend={args.blend} / --gamma={args.gamma}")
        print("[τ Report]")
        print(json.dumps(report, indent=2))
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] saved -> {out_json}")
        return

    # ----- PFN-only (two-head) 경로 -----
    if args.use_ckpt_mu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mu0_pfn, mu1_pfn = load_mu_from_twohead_ckpt(X, CKPT, device=device)

        # paper 모드면 DR/AIPW 타깃 계산에 DML 산출물 필요
        if args.waae_mode == "paper":
            e_path = OUT_DIR / "e_hat.npy"
            if not e_path.exists():
                raise FileNotFoundError("paper mode needs e_hat.npy in --outputs_dir (step 04).")
            e_hat = np.load(e_path)
            mu0_dml = np.load(OUT_DIR / "mu0_hat.npy")
            mu1_dml = np.load(OUT_DIR / "mu1_hat.npy")
            dml_stats = waae_paper(Y, T, mu0_dml, mu1_dml, e_hat, mu0_dml, mu1_dml)
            pfn_stats = waae_paper(Y, T, mu0_pfn, mu1_pfn, e_hat, mu0_dml, mu1_dml)
        else:
            mu0_dml = np.load(OUT_DIR / "mu0_hat.npy")
            mu1_dml = np.load(OUT_DIR / "mu1_hat.npy")
            dml_stats = waae_kpi(Y, T, mu0_dml, mu1_dml)
            pfn_stats = waae_kpi(Y, T, mu0_pfn, mu1_pfn)

        dml_waae, pfn_waae = dml_stats["waae"], pfn_stats["waae"]
        rel_impr = (dml_waae - pfn_waae) / dml_waae if dml_waae > 0 else 0.0
        target_ok = rel_impr >= 0.10

        report = {
            "mode": args.waae_mode,
            "blend": {"mode": "pfn-only"},
            "DML": dml_stats,
            "PFN_only": pfn_stats,
            "relative_improvement": float(rel_impr),
            "target_met_10pct_reduction": bool(target_ok),
            "ckpt_evaluated": str(CKPT),
            "notes": "PFN-only (two-head) evaluation: μ0/μ1 from PFN. Blending ignored."
        }
        print("[WAAE Report]")
        print(json.dumps(report, indent=2))
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[OK] saved -> {out_json}")
        return

    # ----- KPI/PAPER 모드 (PFN+DML) -----
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

    mu0_pfn = mu0_dml
    mu1_pfn_raw = mu0_dml + tau_pfn
    mu1_pfn = mu1_pfn_raw.copy()

    # Load zeta (if available)
    zeta_path = SHIFT_DIR / "05_zeta_bonus.npy"
    if zeta_path.exists():
        print(f"[info] using zeta from: {zeta_path}")
        zeta = np.load(str(zeta_path)).astype(np.float32)
        z = zeta / max(zeta.mean(), 1e-8)
    else:
        print(f"[info] {zeta_path} not found -> NO-SHIFT blending (gamma only or raw).")
        zeta = None
        z = None

    # Orientation check (use T=1 residual correlation)
    mask1 = (T == 1)
    res1 = (Y - mu1_dml)[mask1]
    tau1 = tau_pfn[mask1]
    eps = 1e-8
    corr = float(np.corrcoef(res1, tau1)[0, 1]) if res1.std() > eps and tau1.std() > eps else 0.0
    print("[debug] corr(res1, tau1) on T=1:", corr)
    if corr < 0.0:
        print("[debug] flip tau_pfn sign to align orientation")
        tau_pfn = -tau_pfn
        mu1_pfn_raw = mu0_dml + tau_pfn

    # --------- Blending choices ----------
    if args.blend == "none":
        mu1_pfn = mu1_pfn_raw
        blend_meta = {"mode": "none"}

    elif args.blend == "gamma":
        # γ-블렌드: w = γ * zeta_norm(zeta)
        if zeta is None:
            # NO-SHIFT: 균등 가중
            w = np.full_like(mu1_dml, fill_value=args.gamma, dtype=np.float32)
            zeta_mean_for_meta = None
            zeta_norm_mode = "none"
            zeta_denom_for_meta = None
        else:
            # 정규화 스위치: none / mean / p95
            if args.zeta_norm == "none":
                # w = γ * zeta
                z_base = zeta.astype(np.float32)
                zeta_denom_for_meta = None
                zeta_norm_mode = "none"
            elif args.zeta_norm == "mean":
                # w = γ * zeta / E[zeta]
                denom = float(max(zeta.mean(), 1e-8))
                z_base = (zeta / denom).astype(np.float32)
                zeta_denom_for_meta = denom
                zeta_norm_mode = "mean"
            else:  # "p95"
                # w = γ * zeta / q95(zeta)
                q95 = float(np.quantile(zeta, 0.95))
                denom = max(q95, 1e-8)
                z_base = (zeta / denom).astype(np.float32)
                zeta_denom_for_meta = denom
                zeta_norm_mode = "p95"

            w = (args.gamma * z_base).astype(np.float32)
            zeta_mean_for_meta = float(zeta.mean())

        w = np.clip(w, 0.0, 1.0)
        mu1_pfn = (1.0 - w) * mu1_dml + w * mu1_pfn_raw
        blend_meta = {
            "mode": "gamma",
            "gamma": float(args.gamma),
            "zeta_norm": zeta_norm_mode,
            "zeta_mean": zeta_mean_for_meta,
            "zeta_denom": zeta_denom_for_meta,
            "w_mean": float(w.mean()),
            "w_q95": float(np.quantile(w, 0.95)),
        }

    # elif args.blend == "gamma":
    #     # convex combo with gamma and (optional) zeta, with zeta_norm switch
    #     if zeta is None:
    #         # NO-SHIFT: 균등 가중
    #         w = np.full_like(mu1_dml, fill_value=args.gamma, dtype=np.float32)
    #         zeta_mean_for_meta = None
    #         zeta_norm_mode = "none"
    #     else:
    #         if args.zeta_norm == "none":
    #             # 정규화 끔: w = gamma * zeta (클립만)
    #             w = args.gamma * zeta.astype(np.float32)
    #             zeta_mean_for_meta = float(zeta.mean())
    #             zeta_norm_mode = "none"
    #         else:
    #             # 기본(기존) 동작: w = gamma * (zeta / mean(zeta))
    #             w = args.gamma * z
    #             zeta_mean_for_meta = float(zeta.mean())
    #             zeta_norm_mode = "mean"
    #
    #     w = np.clip(w, 0.0, 1.0).astype(np.float32)
    #     mu1_pfn = (1.0 - w) * mu1_dml + w * mu1_pfn_raw
    #     blend_meta = {
    #         "mode": "gamma",
    #         "gamma": float(args.gamma),
    #         "zeta_mean": zeta_mean_for_meta,
    #         "zeta_norm": zeta_norm_mode,
    #         "w_mean": float(w.mean())
    #     }

    # elif args.blend == "gamma":
    #     if z is None:
    #         w = np.full_like(mu1_dml, fill_value=args.gamma, dtype=np.float32)
    #     else:
    #         w = args.gamma * z
    #     w = np.clip(w, 0.0, 1.0).astype(np.float32)
    #     mu1_pfn = (1.0 - w) * mu1_dml + w * mu1_pfn_raw
    #     blend_meta = {"mode": "gamma", "gamma": float(args.gamma), "zeta_mean": float(zeta.mean()) if zeta is not None else None}
    else:
        # alpha: mean-preserving additive blend
        if z is None:
            tau_w_mean = float(tau_pfn.mean())
            z_desc = "no-zeta"
        else:
            tau_w_mean = float((z * tau_pfn).mean())
            z_desc = f"zeta mean={float(zeta.mean()):.6f}"

        if args.waae_mode == "paper":
            e1 = np.clip(e_hat, 1e-3, 1 - 1e-3)
            ind1 = (T == 1).astype(np.float32)
            aipw1 = mu1_dml + ind1 / e1 * (Y - mu1_dml)
            m1_true = float(aipw1.mean())
        else:
            m1_true = float(Y[mask1].mean()) if mask1.any() else float(Y.mean())
        m1_dml = float(mu1_dml.mean())

        if abs(tau_w_mean) < 1e-10:
            alpha = 0.0
        else:
            alpha = (m1_true - m1_dml) / tau_w_mean
        print(f"[debug] alpha (unconstrained): {alpha} | {z_desc}, tau_w_mean={tau_w_mean:.6f}")

        if z is None:
            mu1_pfn = mu1_dml + alpha * tau_pfn
        else:
            mu1_pfn = mu1_dml + alpha * (z * tau_pfn)

        blend_meta = {"mode": "alpha", "alpha": float(alpha), "zeta_mean": float(zeta.mean()) if zeta is not None else None}

    # --------- Compute metrics ----------
    if args.waae_mode == "paper":
        dml_stats = waae_paper(Y, T, mu0_dml, mu1_dml, e_hat, mu0_dml, mu1_dml)
        pfn_stats = waae_paper(Y, T, mu0_pfn, mu1_pfn, e_hat, mu0_dml, mu1_dml)
    else:  # kpi
        dml_stats = waae_kpi(Y, T, mu0_dml, mu1_dml)
        pfn_stats = waae_kpi(Y, T, mu0_pfn, mu1_pfn)

    dml_waae = dml_stats["waae"]
    pfn_waae = pfn_stats["waae"]
    rel_impr = (dml_waae - pfn_waae) / dml_waae if dml_waae > 0 else 0.0
    target_ok = rel_impr >= 0.10

    # Optional debugging dumps
    aux = {}
    if args.dump_preds:
        np.save(OUT_DIR / f"mu1_pred_{args.blend}.npy", mu1_pfn.astype(np.float32))
        if zeta is not None:
            diff = (mu1_pfn - mu1_dml).astype(np.float32)
            L1 = float(np.mean(np.abs(diff)))
            c = float(np.corrcoef(diff, zeta)[0, 1]) if np.std(diff) > 1e-8 and np.std(zeta) > 1e-8 else 0.0
            aux.update({"L1_mu1diff": L1, "corr_mu1diff_zeta": c})

    report = {
        "mode": args.waae_mode,
        "blend": blend_meta,
        "DML": dml_stats,
        "PFN_plus_DML": pfn_stats,
        "relative_improvement": float(rel_impr),
        "target_met_10pct_reduction": bool(target_ok),
        "ckpt_evaluated": str(CKPT),
        "notes": "paper mode uses DR/AIPW for μ(t). PFN(+DML) uses mu0 from DML; mu1 per chosen blend.",
        "aux": aux
    }

    print("[WAAE Report]")
    print(json.dumps(report, indent=2))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] saved -> {out_json}")


if __name__ == "__main__":
    main()



# import os
# import json
# import argparse
# import numpy as np
# from pathlib import Path
# import torch
#
#
# def find_repo_root(start: Path) -> Path:
#     for p in [start] + list(start.parents):
#         if (p / ".git").exists():
#             return p
#     return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]
#
#
# def load_bonus_arrays(bonus_dir: Path):
#     X = np.load(bonus_dir / "X.npy").astype(np.float32)
#     T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
#     Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
#     return X, T, Y
#
#
# def waae_kpi(Y, T, mu0_pred, mu1_pred):
#     """
#     Project KPI:
#       WAAE = r0 * | mean(mu0 | T=0) - mean(Y | T=0) |
#            + r1 * | mean(mu1 | T=1) - mean(Y | T=1) |
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
#
#
# def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_hat_dml, mu1_hat_dml):
#     """
#     Paper (r69) definition:
#       WAAE(μ̂) = sum_t | μ̂_bar(t) - μ_bar(t) | * P_N(t)
#     DR/AIPW:
#       μ_bar(0) = 1/N * Σ [ μ̂0(Xi) + 1{Ti=0}/(1-ê(Xi)) * (Yi - μ̂0(Xi)) ]
#       μ_bar(1) = 1/N * Σ [ μ̂1(Xi) + 1{Ti=1}/ ê(Xi)    * (Yi - μ̂1(Xi)) ]
#     """
#     r1 = float((T == 1).mean())
#     r0 = 1.0 - r1
#
#     mu0_bar_hat = float(mu0_pred.mean())
#     mu1_bar_hat = float(mu1_pred.mean())
#
#     e1 = np.clip(e_hat, 1e-3, 1 - 1e-3)
#     e0 = 1.0 - e1
#     ind0 = (T == 0).astype(np.float32)
#     ind1 = 1.0 - ind0
#
#     aipw0 = mu0_hat_dml + ind0 / e0 * (Y - mu0_hat_dml)
#     aipw1 = mu1_hat_dml + ind1 / e1 * (Y - mu1_hat_dml)
#     mu0_bar = float(aipw0.mean())
#     mu1_bar = float(aipw1.mean())
#
#     err0 = abs(mu0_bar_hat - mu0_bar)
#     err1 = abs(mu1_bar_hat - mu1_bar)
#     waae = r0 * err0 + r1 * err1
#
#     return dict(
#         waae=float(waae), r0=float(r0), r1=float(r1),
#         err0=float(err0), err1=float(err1),
#         mu0_mean_pred=float(mu0_bar_hat), mu1_mean_pred=float(mu1_bar_hat),
#         mu0_mean_true=float(mu0_bar),     mu1_mean_true=float(mu1_bar)
#     )
#
#
# # (Legacy only) PFN-lite — used only if old checkpoints must be rebuilt
# class PFNLite(torch.nn.Module):
#     def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.0):
#         super().__init__()
#         layers = []
#         d = in_dim
#         for _ in range(depth):
#             layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU(inplace=True)]
#             if dropout > 0:
#                 layers += [torch.nn.Dropout(p=dropout)]
#             d = hidden
#         layers += [torch.nn.Linear(d, 1)]
#         self.net = torch.nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.net(x)  # (n,1)
#
#
# def load_tau_pred_with_ckpt(X, ckpt_path, device="cpu"):
#     """
#     Priority:
#       1) npz with 'tau_pfn'
#       2) torch ckpt with 'tau_pfn'
#       3) (legacy) rebuild PFNLite from cfg/state_dict
#     """
#     n = len(X)
#
#     # 1) npz
#     try:
#         with np.load(ckpt_path) as z:
#             if "tau_pfn" in z:
#                 tau = z["tau_pfn"].astype(np.float32).reshape(-1)
#                 assert len(tau) == n
#                 return tau
#     except Exception:
#         pass
#
#     # 2) torch ckpt
#     ckpt = torch.load(ckpt_path, map_location=device)
#     if isinstance(ckpt, dict) and "tau_pfn" in ckpt:
#         tau = ckpt["tau_pfn"]
#         if torch.is_tensor(tau):
#             tau = tau.detach().cpu().numpy()
#         tau = np.asarray(tau, dtype=np.float32).reshape(-1)
#         assert len(tau) == n
#         return tau
#
#     # 3) legacy rebuild
#     need_keys = {"cfg", "p", "state_dict"}
#     if isinstance(ckpt, dict) and need_keys.issubset(set(ckpt.keys())):
#         cfg = ckpt["cfg"]
#         p = int(ckpt["p"])
#         sd = ckpt["state_dict"]
#         for k in ("hidden", "depth", "dropout"):
#             if k not in cfg:
#                 raise KeyError(f"legacy path: cfg['{k}'] missing")
#
#         model = PFNLite(in_dim=p,
#                         hidden=cfg["hidden"],
#                         depth=cfg["depth"],
#                         dropout=cfg["dropout"]).to(device)
#         model.load_state_dict(sd, strict=False)
#         model.eval()
#         with torch.no_grad():
#             Xt = torch.from_numpy(X.astype(np.float32)).to(device)
#             tau = model(Xt).detach().cpu().numpy().astype(np.float32).reshape(-1)
#         assert len(tau) == n
#         return tau
#
#     raise KeyError(
#         "No 'tau_pfn' found and legacy keys (cfg/p/state_dict) not present. "
#         "Please evaluate with a ckpt that contains 'tau_pfn' or keep the legacy reconstruction path."
#     )
#
#
# def main():
#     ap = argparse.ArgumentParser(
#         description="Evaluate WAAE (KPI or Paper/r69) for DML baseline and PFN(+DML) combo.")
#     ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus",
#                     help="Folder with X.npy/T.npy/Y.npy.")
#     ap.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs",
#                     help="Folder where mu0_hat.npy, mu1_hat.npy, e_hat.npy live; JSON report is stored here by default.")
#     ap.add_argument("--ckpt", type=str, required=True,
#                     help="Path to 06_pfnlite.ckpt to evaluate (e.g., outputs_B/06_pfnlite.ckpt).")
#     ap.add_argument("--out_json", type=str, default=None,
#                     help="Path to save JSON report (default: outputs/07_waae_report.json).")
#     ap.add_argument("--waae_mode", type=str, default="kpi",
#                     choices=["kpi", "paper"],
#                     help="kpi: project KPI definition; paper: r69 WAAE with DR/AIPW.")
#     ap.add_argument("--blend", type=str, default="alpha",
#                     choices=["alpha", "gamma", "none"],
#                     help="alpha: mean-preserving; gamma: convex blend; none: raw mu1 = mu0_dml + tau_pfn.")
#     ap.add_argument("--gamma", type=float, default=0.8,
#                     help="Gamma for 'gamma' blend (w = gamma * zeta / E[zeta]).")
#     ap.add_argument("--dump_preds", action="store_true",
#                     help="If set, dump mu1 predictions and auxiliary metrics for debugging.")
#     ap.add_argument("--shift_outputs_dir", type=str, default=None,
#                     help="Folder to load 05_zeta_bonus.npy from. If None, use outputs_dir. "
#                          "If file is missing, no-shift blending is used.")
#     args = ap.parse_args()
#
#     # Resolve paths
#     here = Path(__file__).resolve()
#     repo_root = find_repo_root(here)
#     BONUS_DIR = (repo_root / args.bonus_dir).resolve()
#     OUT_DIR = (repo_root / args.outputs_dir).resolve()
#     CKPT = (repo_root / args.ckpt).resolve()
#     OUT_DIR.mkdir(parents=True, exist_ok=True)
#     out_json = (repo_root / args.out_json).resolve() if args.out_json else (OUT_DIR / "07_waae_report.json")
#     SHIFT_DIR = (repo_root / (args.shift_outputs_dir or args.outputs_dir)).resolve()
#
#     # Load arrays
#     X, T, Y = load_bonus_arrays(BONUS_DIR)
#
#     # Load DML artifacts
#     mu0_dml = np.load(OUT_DIR / "mu0_hat.npy")
#     mu1_dml = np.load(OUT_DIR / "mu1_hat.npy")
#     if args.waae_mode == "paper":
#         e_path = OUT_DIR / "e_hat.npy"
#         if not e_path.exists():
#             raise FileNotFoundError(f"Missing {e_path}. Save e_hat.npy in step 04 or switch to --waae_mode kpi.")
#         e_hat = np.load(e_path)
#     else:
#         e_hat = None
#
#     # Load PFN tau
#     tau_pfn = load_tau_pred_with_ckpt(X, CKPT)
#     mu0_pfn = mu0_dml
#     mu1_pfn = mu0_dml + tau_pfn
#     mu1_pfn_raw = mu1_pfn.copy()
#
#     # --- DEBUG 1: PFN signal magnitude ---
#     print("[debug] tau_pfn mean/std:", float(np.mean(tau_pfn)), float(np.std(tau_pfn)))
#
#     # Load zeta (if available)
#     # zeta_path = OUT_DIR / "05_zeta_bonus.npy"
#     zeta_path = SHIFT_DIR / "05_zeta_bonus.npy"
#
#     if zeta_path.exists():
#         print(f"[info] using zeta from: {zeta_path}")
#         zeta = np.load(str(zeta_path)).astype(np.float32)
#     else:
#         print(f"[info] {zeta_path} not found -> NO-SHIFT blending (gamma only or raw).")
#         zeta = None
#
#     if zeta is None:
#         print(f"[info] zeta not found at {zeta_path} -> no-shift weighting.")
#         z = None
#     else:
#         z = zeta / max(zeta.mean(), 1e-8)
#
#     # Orientation check (use T=1 residual correlation)
#     mask1 = (T == 1)
#     res1 = (Y - mu1_dml)[mask1]
#     tau1 = tau_pfn[mask1]
#     eps = 1e-8
#     corr = float(np.corrcoef(res1, tau1)[0, 1]) if res1.std() > eps and tau1.std() > eps else 0.0
#     print("[debug] corr(res1, tau1) on T=1:", corr)
#     if corr < 0.0:
#         print("[debug] flip tau_pfn sign to align orientation")
#         tau_pfn = -tau_pfn
#         mu1_pfn_raw = mu0_dml + tau_pfn
#         if z is not None:
#             tau1 = -tau1  # for completeness
#
#     # --------- Blending choices ----------
#     if args.blend == "none":
#         # raw: keep mu1_pfn_raw
#         mu1_pfn = mu1_pfn_raw
#         blend_meta = {"mode": "none"}
#     elif args.blend == "gamma":
#         # convex combo with gamma and (optional) zeta
#         if z is None:
#             w = np.full_like(mu1_dml, fill_value=args.gamma, dtype=np.float32)
#         else:
#             w = args.gamma * z
#         w = np.clip(w, 0.0, 1.0).astype(np.float32)
#         mu1_pfn = (1.0 - w) * mu1_dml + w * mu1_pfn_raw
#         blend_meta = {"mode": "gamma", "gamma": float(args.gamma), "zeta_mean": float(zeta.mean()) if zeta is not None else None}
#     else:
#         # alpha: mean-preserving additive blend -> enforce global mean(μ1) to DR target (paper mode) or obs mean (kpi)
#         if z is None:
#             # mean-preserving without z: use plain tau mean
#             tau_w_mean = float(tau_pfn.mean())
#             z_desc = "no-zeta"
#         else:
#             tau_w_mean = float((z * tau_pfn).mean())
#             z_desc = f"zeta mean={float(zeta.mean()):.6f}"
#
#         if args.waae_mode == "paper":
#             e1 = np.clip(e_hat, 1e-3, 1 - 1e-3)
#             ind1 = (T == 1).astype(np.float32)
#             aipw1 = mu1_dml + ind1 / e1 * (Y - mu1_dml)
#             m1_true = float(aipw1.mean())
#         else:
#             m1_true = float(Y[mask1].mean()) if mask1.any() else float(Y.mean())
#         m1_dml = float(mu1_dml.mean())
#
#         if abs(tau_w_mean) < 1e-10:
#             alpha = 0.0
#         else:
#             alpha = (m1_true - m1_dml) / tau_w_mean  # unconstrained
#         print(f"[debug] alpha (unconstrained): {alpha} | {z_desc}, tau_w_mean={tau_w_mean:.6f}")
#
#         if z is None:
#             mu1_pfn = mu1_dml + alpha * tau_pfn
#         else:
#             mu1_pfn = mu1_dml + alpha * (z * tau_pfn)
#
#         blend_meta = {"mode": "alpha", "alpha": float(alpha), "zeta_mean": float(zeta.mean()) if zeta is not None else None}
#
#     # --------- Compute metrics ----------
#     if args.waae_mode == "paper":
#         dml_stats = waae_paper(Y, T, mu0_dml, mu1_dml, e_hat, mu0_dml, mu1_dml)
#         pfn_stats = waae_paper(Y, T, mu0_pfn, mu1_pfn, e_hat, mu0_dml, mu1_dml)
#     else:
#         dml_stats = waae_kpi(Y, T, mu0_dml, mu1_dml)
#         pfn_stats = waae_kpi(Y, T, mu0_pfn, mu1_pfn)
#
#     dml_waae = dml_stats["waae"]
#     pfn_waae = pfn_stats["waae"]
#     rel_impr = (dml_waae - pfn_waae) / dml_waae if dml_waae > 0 else 0.0
#     target_ok = rel_impr >= 0.10
#
#     # Optional debugging dumps
#     aux = {}
#     if args.dump_preds:
#         np.save(OUT_DIR / f"mu1_pred_{args.blend}.npy", mu1_pfn.astype(np.float32))
#         if zeta is not None:
#             diff = (mu1_pfn - mu1_dml).astype(np.float32)
#             L1 = float(np.mean(np.abs(diff)))
#             c = float(np.corrcoef(diff, zeta)[0, 1]) if np.std(diff) > 1e-8 and np.std(zeta) > 1e-8 else 0.0
#             aux.update({"L1_mu1diff": L1, "corr_mu1diff_zeta": c})
#
#     report = {
#         "mode": args.waae_mode,
#         "blend": blend_meta,
#         "DML": dml_stats,
#         "PFN_plus_DML": pfn_stats,
#         "relative_improvement": float(rel_impr),
#         "target_met_10pct_reduction": bool(target_ok),
#         "ckpt_evaluated": str(CKPT),
#         "notes": "paper mode uses DR/AIPW for μ(t). PFN(+DML) uses mu0 from DML; mu1 per chosen blend.",
#         "aux": aux
#     }
#
#     print("[WAAE Report]")
#     print(json.dumps(report, indent=2))
#     with open(out_json, "w") as f:
#         json.dump(report, f, indent=2)
#     print(f"[OK] saved -> {out_json}")
#
#
# if __name__ == "__main__":
#     main()
