# benchmarks/bonus/06_pfn_cate_blend.py
# ---------------------------------------------------------------------
# Method-1 (using ORIGINAL CausalPFN from vdblm/CausalPFN):
#   1) Get CATE (tau) on BONUS via CATEEstimator.fit(...); estimate_cate(...)
#   2) Build PFN+anchor mu1 via mu1_raw = mu0_DML + tau_PFN
#   3) Shift-guided blending: mu1_blend = (1-w)*mu1_DML + w*mu1_raw,
#      with w_x = gamma * zeta / mean(zeta)
#   4) Save as 06_pfnlite.ckpt (npz) compatible with your 07 script
#
# API reference: README Quick Start shows CATEEstimator.fit / estimate_cate
#   https://github.com/vdblm/CausalPFN  (see README lines with CATEEstimator)
# ---------------------------------------------------------------------

# python benchmarks/bonus/06_pfn_cate_blend.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_outputs_dir benchmarks/bonus/outputs \
#   --shift_outputs_dir benchmarks/bonus/outputs \
#   --outputs_dir benchmarks/bonus/outputs_CPFN_official \

# python benchmarks/bonus/06_pfn_cate_blend.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_outputs_dir benchmarks/bonus/outputs \
#   --shift_outputs_dir benchmarks/bonus/outputs \
#   --outputs_dir benchmarks/bonus/outputs_CPFN_official_noshift \
# #


# no shift
# python benchmarks/bonus/06_pfn_cate_blend.py \
#   --mode pfn_plus_dml \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_outputs_dir benchmarks/bonus/outputs \
#   --shift_outputs_dir benchmarks/bonus/outputs_noshift \
#   --outputs_dir benchmarks/bonus/outputs_CPFN_official_noshift \
#   --gamma 0.0

# shift
# python benchmarks/bonus/06_pfn_cate_blend.py \
#   --mode pfn_plus_dml
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_outputs_dir benchmarks/bonus/outputs \
#   --shift_outputs_dir benchmarks/bonus/outputs \
#   --outputs_dir benchmarks/bonus/outputs_CPFN_official \
#   --gamma 0.8

# # pfn only
# python benchmarks/bonus/06_pfn_cate_blend.py \
#   --mode pfn_only \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --outputs_dir benchmarks/bonus/outputs_CPFN_only

## DML 을 outputs mlp05로 씀
# python benchmarks/bonus/06_pfn_cate_blend.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_outputs_dir benchmarks/bonus/outputs_mlp05 \
#   --shift_outputs_dir benchmarks/bonus/outputs_mlp05 \
#   --outputs_dir benchmarks/bonus/outputs_CPFN_tau \
#   --gamma 0.8




# benchmarks/bonus/06_pfn_cate_blend.py
# ---------------------------------------------------------------------
# Method-1 (using ORIGINAL CausalPFN from vdblm/CausalPFN):
#   1) Get CATE (tau) on BONUS via CATEEstimator.fit(...); estimate_cate(...)
#   2) Build PFN+anchor mu1 via mu1_raw = mu0_DML + tau_PFN
#   3) Shift-guided blending: mu1_blend = (1-w)*mu1_DML + w*mu1_raw,
#      with w_x = gamma * zeta / mean(zeta)
#   4) Save as 06_pfnlite.ckpt (torch .pt) compatible with your 07 script
# ---------------------------------------------------------------------
# benchmarks/bonus/06_pfn_cate_blend.py
# ---------------------------------------------------------------------
# Two modes in one script:
#   (A) --mode pfn_only
#       - Use vdblm/CausalPFN CATEEstimator to get tau(x) only.
#       - Save ckpt with 'tau_pfn' (no mu0/mu1), for τ-based eval later.
#   (B) --mode pfn_plus_dml   [default]
#       - Anchor on DML: mu1_raw = mu0_dml + tau_pfn
#       - Optional shift weighting: w = gamma * zeta / E[zeta] (clipped to [0,1])
#       - Save mu1_blend etc. so 07 WAAE(paper) can be computed.
# ---------------------------------------------------------------------
import argparse, json
from pathlib import Path
import numpy as np

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_bonus(bonus_dir: Path):
    X = np.load(bonus_dir / "X.npy").astype(np.float32)
    T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
    return X, T, Y

def load_dml(outputs_dir: Path):
    mu0 = np.load(outputs_dir / "mu0_hat.npy").astype(np.float32)
    mu1 = np.load(outputs_dir / "mu1_hat.npy").astype(np.float32)
    ehat = np.load(outputs_dir / "e_hat.npy").astype(np.float32)
    return mu0, mu1, ehat

def load_zeta(outputs_dir: Path):
    p = outputs_dir / "05_zeta_bonus.npy"
    if not p.exists():
        print(f"[info] zeta not found at {p} -> no-shift mode for blending.")
        return None
    return np.load(p).astype(np.float32)

def get_tau_with_causalpfn_official(X, T, Y, device=None, verbose=True):
    try:
        import torch
    except Exception:
        torch = None
    dev = device
    if dev is None and torch is not None:
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev is None:
        dev = "cpu"
    from causalpfn import CATEEstimator
    est = CATEEstimator(device=dev, verbose=verbose)
    est.fit(X, T, Y)
    tau = est.estimate_cate(X)  # (n,)
    tau = np.asarray(tau, dtype=np.float32).reshape(-1)
    assert tau.shape[0] == X.shape[0]
    return tau, dev

def main():
    ap = argparse.ArgumentParser("06: CausalPFN CATE + (optional) DML anchor/shift blend")
    ap.add_argument("--mode", type=str, default="pfn_plus_dml",
                    choices=["pfn_plus_dml", "pfn_only"],
                    help="pfn_only: save tau only (PFN-only evaluation later). pfn_plus_dml: build mu1 with DML anchor (+shift/gamma).")
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus")
    ap.add_argument("--dml_outputs_dir", type=str, default="benchmarks/bonus/outputs")
    ap.add_argument("--shift_outputs_dir", type=str, default="benchmarks/bonus/outputs")
    ap.add_argument("--outputs_dir", type=str, required=True)
    ap.add_argument("--gamma", type=float, default=0.8, help="Only for pfn_plus_dml: global PFN mixing in [0,1].")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    BONUS_DIR = (repo_root / args.bonus_dir).resolve()
    DML_DIR   = (repo_root / args.dml_outputs_dir).resolve()
    SHIFT_DIR = (repo_root / args.shift_outputs_dir).resolve()
    OUT_DIR   = (repo_root / args.outputs_dir).resolve()
    ensure_dir(OUT_DIR)

    X, T, Y = load_bonus(BONUS_DIR)
    n = len(X)

    # 1) 항상 PFN으로 tau는 뽑는다
    tau_pfn, dev = get_tau_with_causalpfn_official(X, T, Y, device=args.device, verbose=True)
    print(f"[info] device={dev} | tau_pfn mean/std = {tau_pfn.mean():.6f}/{tau_pfn.std():.6f}")

    import torch
    if args.mode == "pfn_only":
        # 2A) PFN-only: tau만 저장 (mu0/mu1 없음). 07에서 τ-평가 모드로 쓰게 됨.
        ckpt = {
            "tau_pfn": torch.from_numpy(tau_pfn.astype(np.float32)),
            "p": int(X.shape[1]),
            "cfg": {"format": "torch", "source": "PFN-only (tau only) via CATEEstimator"}
        }
        torch.save(ckpt, OUT_DIR / "06_pfnlite.ckpt")
        meta = {"mode": "pfn_only", "bonus_dir": str(BONUS_DIR), "outputs_dir": str(OUT_DIR)}
        with open(OUT_DIR / "06_meta.json", "w") as f: json.dump(meta, f, indent=2)
        print(f"[OK] saved ckpt -> {OUT_DIR / '06_pfnlite.ckpt'}")
        return

    # 2B) PFN + DML: DML 앵커 로드
    mu0_dml, mu1_dml, e_hat = load_dml(DML_DIR)
    assert len(mu0_dml) == n and len(mu1_dml) == n and len(e_hat) == n
    zeta = load_zeta(SHIFT_DIR)
    if zeta is not None: assert len(zeta) == n

    print(f"[info] DML  : mu0/mu1={mu0_dml.shape}/{mu1_dml.shape}, e_hat={e_hat.shape}")
    if zeta is None:
        print("[info] shift: zeta=None (NO-SHIFT)")
        mean_zeta = 0.0
        w = np.zeros(n, dtype=np.float32) if args.gamma == 0.0 else np.full(n, fill_value=args.gamma, dtype=np.float32)
    else:
        mean_zeta = float(max(zeta.mean(), 1e-8))
        w = args.gamma * (zeta / mean_zeta)
        w = np.clip(w, 0.0, 1.0).astype(np.float32)
        print(f"[info] shift: zeta mean={zeta.mean():.6f} | gamma={args.gamma} | mean(w)={w.mean():.6f}")

    mu1_raw   = mu0_dml + tau_pfn
    mu1_blend = (1.0 - w) * mu1_dml + w * mu1_raw

    ckpt = {
        "tau_pfn": torch.from_numpy(tau_pfn.astype(np.float32)),
        "mu1_raw": torch.from_numpy(mu1_raw.astype(np.float32)),
        "mu1_blend": torch.from_numpy(mu1_blend.astype(np.float32)),
        "mu0_dml": torch.from_numpy(mu0_dml.astype(np.float32)),
        "mu1_dml": torch.from_numpy(mu1_dml.astype(np.float32)),
        "e_hat": torch.from_numpy(e_hat.astype(np.float32)),
        "zeta": torch.from_numpy((zeta if zeta is not None else np.zeros(n, np.float32))),
        "gamma": float(args.gamma),
        "mean_zeta": float(mean_zeta),
        "p": int(X.shape[1]),
        "cfg": {
            "format": "torch",
            "source": "PFN+DML anchor with (gamma*zeta) blending",
            "gamma": float(args.gamma),
            "mean_zeta": float(mean_zeta),
        },
    }
    torch.save(ckpt, OUT_DIR / "06_pfnlite.ckpt")
    meta = {
        "mode": "pfn_plus_dml",
        "bonus_dir": str(BONUS_DIR),
        "dml_outputs_dir": str(DML_DIR),
        "shift_outputs_dir": str(SHIFT_DIR),
        "outputs_dir": str(OUT_DIR),
        "gamma": args.gamma,
        "mean_zeta": mean_zeta,
    }
    with open(OUT_DIR / "06_meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"[OK] saved ckpt -> {OUT_DIR / '06_pfnlite.ckpt'}")

if __name__ == "__main__":
    main()

# import os
# import json
# import argparse
# from pathlib import Path
# import numpy as np
#
# def find_repo_root(start: Path) -> Path:
#     for p in [start] + list(start.parents):
#         if (p / ".git").exists():
#             return p
#     return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]
#
# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)
#
# def load_bonus(bonus_dir: Path):
#     X = np.load(bonus_dir / "X.npy").astype(np.float32)
#     T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
#     Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
#     return X, T, Y
#
# def load_dml(outputs_dir: Path):
#     mu0 = np.load(outputs_dir / "mu0_hat.npy").astype(np.float32)
#     mu1 = np.load(outputs_dir / "mu1_hat.npy").astype(np.float32)
#     ehat = np.load(outputs_dir / "e_hat.npy").astype(np.float32)
#     return mu0, mu1, ehat
#
# def load_zeta(outputs_dir: Path):
#     """Return zeta if exists; else return None (no-shift mode)."""
#     p = outputs_dir / "05_zeta_bonus.npy"
#     if not p.exists():
#         print(f"[info] zeta not found at {p} -> no-shift mode for blending.")  # ★ 변경
#         return None
#     return np.load(p).astype(np.float32)
#
# def get_tau_with_causalpfn_official(X, T, Y, device=None, verbose=True):
#     """
#     Use the official CausalPFN API per README:
#       est = CATEEstimator(device=..., verbose=...)
#       est.fit(X, T, Y)
#       tau = est.estimate_cate(X)
#     """
#     try:
#         import torch
#     except Exception:
#         torch = None
#
#     dev = device
#     if dev is None and torch is not None:
#         dev = "cuda:0" if torch.cuda.is_available() else "cpu"
#     if dev is None:
#         dev = "cpu"
#
#     from causalpfn import CATEEstimator  # official package API
#     est = CATEEstimator(device=dev, verbose=verbose)
#     est.fit(X, T, Y)
#     tau = est.estimate_cate(X)  # shape (n,)
#     tau = np.asarray(tau, dtype=np.float32).reshape(-1)
#     assert tau.shape[0] == X.shape[0], "tau length mismatch."
#     return tau, dev
#
# def main():
#     ap = argparse.ArgumentParser(
#         description="06 (official CausalPFN): get tau via CATEEstimator.fit/estimate_cate, then DML+shift blending."
#     )
#     ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus")
#     ap.add_argument("--dml_outputs_dir", type=str, default="benchmarks/bonus/outputs")
#     ap.add_argument("--shift_outputs_dir", type=str, default="benchmarks/bonus/outputs")
#     ap.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs_CPFN_official")
#     ap.add_argument("--gamma", type=float, default=0.8, help="Global PFN mixing knob in blending (0..1).")
#     ap.add_argument("--device", type=str, default=None, help="cpu / cuda:0 etc. If None, auto-pick.")
#     args = ap.parse_args()
#
#     here = Path(__file__).resolve()
#     repo_root = find_repo_root(here)
#     BONUS_DIR = (repo_root / args.bonus_dir).resolve()
#     DML_DIR = (repo_root / args.dml_outputs_dir).resolve()
#     SHIFT_DIR = (repo_root / args.shift_outputs_dir).resolve()
#     OUT_DIR = (repo_root / args.outputs_dir).resolve()
#     ensure_dir(OUT_DIR)
#
#     # 1) Load data & artifacts
#     X, T, Y = load_bonus(BONUS_DIR)
#     mu0_dml, mu1_dml, e_hat = load_dml(DML_DIR)
#     zeta = load_zeta(SHIFT_DIR)
#
#     n = len(X)
#     assert len(mu0_dml) == n and len(mu1_dml) == n and len(e_hat) == n, "Length mismatch in DML artifacts."
#     if zeta is not None:
#         assert len(zeta) == n, "Length mismatch in zeta."
#
#     print(f"[info] BONUS: X={X.shape}, T={T.shape}, Y={Y.shape}")
#     print(f"[info] DML  : mu0/mu1={mu0_dml.shape}/{mu1_dml.shape}, e_hat={e_hat.shape}")
#     if zeta is None:
#         print("[info] shift: zeta=None (no-shift)")
#     else:
#         print(f"[info] shift: zeta={zeta.shape} (mean={zeta.mean():.6f})")
#
#     # 2) Get tau from official CausalPFN
#     tau_pfn, dev = get_tau_with_causalpfn_official(X, T, Y, device=args.device, verbose=True)
#     print(f"[info] device={dev} | tau_pfn: mean={tau_pfn.mean():.6f}, std={tau_pfn.std():.6f}")
#
#     # 3) Raw PFN-combined mu1 via anchor:
#     mu1_raw = mu0_dml + tau_pfn
#
#     # 4) Shift-guided blending with gamma (with robust no-shift fallback)  # ★ 변경
#     if (zeta is None) or (args.gamma == 0.0):
#         # no-shift: PFN 영향 0 -> mu1_blend = mu1_dml
#         w = np.zeros(n, dtype=np.float32)
#         print(f"[info] blending: no-shift mode (gamma={args.gamma}); w mean=0.0")
#         zeta_for_save = np.zeros(n, dtype=np.float32)
#     else:
#         mean_zeta = float(max(zeta.mean(), 1e-8))
#         w = float(args.gamma) * (zeta / mean_zeta)
#         w = np.clip(w, 0.0, 1.0).astype(np.float32)
#         print(f"[info] blending: shift mode (gamma={args.gamma}); w mean={w.mean():.6f}")
#         zeta_for_save = zeta.astype(np.float32)
#
#     mu1_blend = (1.0 - w) * mu1_dml + w * mu1_raw
#
#     # 5) Save (torch ckpt compatible with 07 script)
#     import torch
#     out_ckpt = OUT_DIR / "06_pfnlite.ckpt"
#     ckpt_dict = {
#         "tau_pfn": torch.from_numpy(tau_pfn.astype(np.float32)),
#         "mu1_raw": torch.from_numpy(mu1_raw.astype(np.float32)),
#         "mu1_blend": torch.from_numpy(mu1_blend.astype(np.float32)),
#         "mu0_dml": torch.from_numpy(mu0_dml.astype(np.float32)),
#         "mu1_dml": torch.from_numpy(mu1_dml.astype(np.float32)),
#         "e_hat": torch.from_numpy(e_hat.astype(np.float32)),
#         "zeta": torch.from_numpy(zeta_for_save),   # ★ 변경: None 대비
#         "gamma": float(args.gamma),
#         "mean_zeta": float(float(zeta.mean()) if zeta is not None else 0.0),  # ★ 변경
#         "p": int(X.shape[1]),
#         "cfg": {
#             "format": "torch",
#             "source": "CausalPFN_official + DML anchor + (gamma*zeta) blending (no-shift safe)",
#             "gamma": float(args.gamma),
#             "mean_zeta": float(float(zeta.mean()) if zeta is not None else 0.0),  # ★ 변경
#         },
#     }
#     torch.save(ckpt_dict, out_ckpt)
#
#     meta = {
#         "bonus_dir": str(BONUS_DIR),
#         "dml_outputs_dir": str(DML_DIR),
#         "shift_outputs_dir": str(SHIFT_DIR),
#         "outputs_dir": str(OUT_DIR),
#         "gamma": args.gamma,
#         "mean_zeta": (float(zeta.mean()) if zeta is not None else 0.0),
#         "note": "Official CausalPFN (CATEEstimator.fit/estimate_cate) + DML anchor + (gamma*zeta) blending. Safe when zeta is missing.",
#     }
#     with open(OUT_DIR / "06_meta.json", "w") as f:
#         json.dump(meta, f, indent=2)
#
#     print(f"[OK] saved ckpt (torch) -> {out_ckpt}")
#     print(f"[OK] saved meta -> {OUT_DIR / '06_meta.json'}")
#
# if __name__ == "__main__":
#     main()
