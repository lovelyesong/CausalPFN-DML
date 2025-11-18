# benchmarks/bonus/07_eval_waae_multi.py

# no shift
# python benchmarks/bonus/07_eval_waae_multi.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_root benchmarks/bonus/outputs_mlp05 \
#   --ckpt benchmarks/bonus/outputs_CPFN_only/06_pfnlite.ckpt \
#   --waae_mode paper \
#   --blend gamma --gamma 0.8 --zeta_norm mean \
#   --shift_outputs_dir benchmarks/bonus/outputs_mlp05 \
#   --out_json benchmarks/bonus/outputs_mlp05/07_waae_multi.json

# shift
# python benchmarks/bonus/07_eval_waae_multi.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_root benchmarks/bonus/outputs_mlp05 \
#   --ckpt benchmarks/bonus/outputs_CPFN_only/06_pfnlite.ckpt \
#   --waae_mode paper \
#   --blend gamma --gamma 0.8 --zeta_norm p95 \
#   --shift_outputs_dir benchmarks/bonus/outputs_mlp05 \
#   --out_json benchmarks/bonus/outputs_mlp05/07_waae_multi_shift.json

# python benchmarks/bonus/07_eval_waae_multi.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_root benchmarks/bonus/outputs_mlp05 \
#   --ckpt benchmarks/bonus/outputs_CPFN_only/06_pfnlite.ckpt \
#   --waae_mode paper --blend alpha --gamma 0.8 --zeta_norm p95\
#   --shift_outputs_dir benchmarks/bonus/outputs_mlp05 \
#   --out_json benchmarks/bonus/outputs_mlp05/07_waae_multi_alpha_p95.json


# python benchmarks/bonus/07_eval_waae_multi.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --dml_root benchmarks/bonus/outputs_mlp05 \
#   --ckpt benchmarks/bonus/outputs_CPFN_only/06_pfnlite.ckpt \
#   --shift_outputs_dir benchmarks/bonus/outputs_mlp05 \
#   --waae_mode paper \
#   --grid \
#   --gammas 0.0,0.2,0.4,0.6,0.8,1.0 \
#   --zeta_norms mean,p95,none \
#   --blend_modes gamma,alpha \
#   --out_dir benchmarks/bonus/outputs_mlp05/grid_waae

# --bonus_dir
# notebooks/bonus_benchmarks/data/bonus
# --dml_root
# benchmarks/bonus/outputs_dml05
# --ckpt
# benchmarks/bonus/outputs_CPFN_only/06_pfnlite.ckpt
# --shift_outputs_dir
# benchmarks/bonus/outputs_dml05
# --waae_mode
# paper
# --grid
# --gammas
# 0.0,0.2,0.4,0.6,0.8,1.0
# --zeta_norms
# mean,p95,none
# --blend_modes
# gamma,alpha
# --out_dir
# benchmarks/bonus/outputs_dml05/grid_waae


# benchmarks/bonus/07_eval_waae_multi.py
import os, json, argparse, glob
from pathlib import Path
import numpy as np
import torch

def load_bonus_arrays(bonus_dir: Path):
    X = np.load(bonus_dir / "X.npy").astype(np.float32)
    T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
    return X, T, Y

def waae_kpi(Y, T, mu0_pred, mu1_pred):
    m0 = (T==0); m1 = (T==1)
    r0 = float(m0.mean()); r1 = 1.0 - r0
    y0 = float(Y[m0].mean()) if m0.any() else 0.0
    y1 = float(Y[m1].mean()) if m1.any() else 0.0
    e0 = abs(float(mu0_pred[m0].mean()) - y0) if m0.any() else 0.0
    e1 = abs(float(mu1_pred[m1].mean()) - y1) if m1.any() else 0.0
    return r0*e0 + r1*e1

def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_anchor, mu1_anchor):
    r1 = float((T==1).mean()); r0 = 1.0 - r1
    mu0_bar_hat = float(mu0_pred.mean()); mu1_bar_hat = float(mu1_pred.mean())
    e1 = np.clip(e_hat, 1e-3, 1-1e-3); e0 = 1.0 - e1
    ind1 = (T==1).astype(np.float32); ind0 = 1.0 - ind1
    aipw0 = mu0_anchor + ind0/e0 * (Y - mu0_anchor)
    aipw1 = mu1_anchor + ind1/e1 * (Y - mu1_anchor)
    mu0_bar = float(aipw0.mean()); mu1_bar = float(aipw1.mean())
    err0 = abs(mu0_bar_hat - mu0_bar); err1 = abs(mu1_bar_hat - mu1_bar)
    return r0*err0 + r1*err1

def load_tau_from_ckpt(X, ckpt_path, device="cpu"):
    n = len(X)
    try:
        with np.load(ckpt_path) as z:
            if "tau_pfn" in z:
                tau = z["tau_pfn"].astype(np.float32).reshape(-1)
                assert len(tau) == n
                return tau
    except Exception:
        pass
    ckpt = torch.load(ckpt_path, map_location=device)
    tau = ckpt.get("tau_pfn", None)
    if tau is None:
        raise KeyError("ckpt has no 'tau_pfn'")
    if torch.is_tensor(tau):
        tau = tau.detach().cpu().numpy()
    tau = tau.astype(np.float32).reshape(-1)
    assert len(tau) == n
    return tau

def eval_one_seed(bonus_dir: Path, dml_dir: Path, ckpt: Path,
                  waae_mode: str, blend: str, gamma: float,
                  zeta_path: Path|None, zeta_norm: str):
    X, T, Y = load_bonus_arrays(bonus_dir)
    mu0 = np.load(dml_dir / "mu0_hat.npy")
    mu1 = np.load(dml_dir / "mu1_hat.npy")
    e   = np.load(dml_dir / "e_hat.npy")

    # PFN(+DML)
    tau_pfn = load_tau_from_ckpt(X, ckpt)
    mu0_pfn = mu0.copy()
    mu1_raw = mu0 + tau_pfn
    mu1_pfn = mu1_raw.copy()

    # zeta
    zeta = None; z = None
    if zeta_path is not None and zeta_path.exists():
        zeta = np.load(str(zeta_path)).astype(np.float32)
        if zeta_norm == "mean":
            z = zeta / max(float(zeta.mean()), 1e-8)
        elif zeta_norm == "p95":
            q = float(np.quantile(zeta, 0.95))
            z = zeta / max(q, 1e-8)
        else:
            z = zeta.copy()

    # blend
    if blend == "none":
        mu1_pfn = mu1_raw
    elif blend == "gamma":
        if z is None: w = np.full_like(mu1, gamma, dtype=np.float32)
        else:         w = np.clip(gamma * z, 0.0, 1.0).astype(np.float32)
        mu1_pfn = (1.0 - w) * mu1 + w * mu1_raw
    else:  # alpha
        if z is None: tau_w_mean = float(tau_pfn.mean())
        else:         tau_w_mean = float((z * tau_pfn).mean())
        if waae_mode == "paper":
            e1 = np.clip(e, 1e-3, 1-1e-3); ind1 = (T==1).astype(np.float32)
            aipw1 = mu1 + ind1/e1 * (Y - mu1)
            m1_true = float(aipw1.mean())
        else:
            mask1 = (T==1)
            m1_true = float(Y[mask1].mean()) if mask1.any() else float(Y.mean())
        m1_dml = float(mu1.mean())
        alpha = 0.0 if abs(tau_w_mean) < 1e-10 else (m1_true - m1_dml) / tau_w_mean
        mu1_pfn = mu1 + alpha * (tau_pfn if z is None else (z * tau_pfn))

    # metrics
    if waae_mode == "paper":
        dml_waae = waae_paper(Y, T, mu0, mu1, e, mu0, mu1)
        pfn_waae = waae_paper(Y, T, mu0_pfn, mu1_pfn, e, mu0, mu1)
    else:
        dml_waae = waae_kpi(Y, T, mu0, mu1)
        pfn_waae = waae_kpi(Y, T, mu0_pfn, mu1_pfn)

    return dict(DML=float(dml_waae), PFN_plus_DML=float(pfn_waae))

def run_setting(bonus_dir: Path, dml_root: Path, ckpt: Path,
                shift_outputs_dir: Path|None, waae_mode: str,
                blend: str, gamma: float, zeta_norm: str,
                out_json: Path):
    """seed_* 전부 돌려서 aggregate 저장"""
    zeta_path = None
    if shift_outputs_dir:
        zeta_path = Path(shift_outputs_dir) / "05_zeta_bonus.npy"

    seed_dirs = sorted([Path(p) for p in glob.glob(str(dml_root / "seed_*")) if Path(p).is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs under {dml_root}")

    rows = []
    for sd in seed_dirs:
        r = eval_one_seed(
            bonus_dir=bonus_dir, dml_dir=sd, ckpt=ckpt,
            waae_mode=waae_mode, blend=blend, gamma=gamma,
            zeta_path=zeta_path, zeta_norm=zeta_norm
        )
        rows.append({"seed_dir": sd.name, **r})
        print(f"[{sd.name}] DML={r['DML']:.6f} | PFN+DML={r['PFN_plus_DML']:.6f}")

    dml_list  = [x["DML"] for x in rows]
    pfn_list  = [x["PFN_plus_DML"] for x in rows]
    agg = {
        "DML_mean": float(np.mean(dml_list)),
        "DML_std":  float(np.std(dml_list, ddof=1)) if len(dml_list)>1 else 0.0,
        "PFN_mean": float(np.mean(pfn_list)),
        "PFN_std":  float(np.std(pfn_list, ddof=1)) if len(pfn_list)>1 else 0.0,
        "relative_improvement_mean": float(np.mean([(d-p)/d if d>0 else 0.0 for d,p in zip(dml_list,pfn_list)])),
        "n_runs": len(rows)
    }
    report = {
        "mode": waae_mode,
        "blend": blend,
        "gamma": float(gamma),
        "zeta_norm": zeta_norm,
        "rows": rows,
        "aggregate": agg,
        "paths": {
            "dml_root": str(dml_root),
            "ckpt": str(ckpt),
            "zeta_path": str(zeta_path) if zeta_path is not None else None
        }
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] saved -> {out_json}")
    print(f"[SUMMARY] DML mean={agg['DML_mean']:.6f} (±{agg['DML_std']:.6f}), "
          f"PFN+DML mean={agg['PFN_mean']:.6f} (±{agg['PFN_std']:.6f}), "
          f"Rel.Impr mean={agg['relative_improvement_mean']:.3%}")
    return {"out_json": str(out_json), **report["aggregate"]}

def parse_list_floats(s: str):
    return [float(x) for x in s.split(",")] if s else []

def parse_list_str(s: str):
    return [x.strip() for x in s.split(",")] if s else []

def main():
    ap = argparse.ArgumentParser("Evaluate WAAE across multiple seed_* dirs (single or grid)")
    # 공통
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus")
    ap.add_argument("--dml_root", type=str, required=True, help="05 결과 루트 (seed_* 하위에 mu0/mu1/e_hat.npy 존재)")
    ap.add_argument("--ckpt", type=str, required=True, help="06_pfnlite.ckpt (tau_pfn 포함)")
    ap.add_argument("--shift_outputs_dir", type=str, default=None, help="zeta 위치(05_zeta_bonus.npy)")
    ap.add_argument("--waae_mode", type=str, default="paper", choices=["paper","kpi"])

    # 단일 실행(기존)
    ap.add_argument("--blend", type=str, default="gamma", choices=["gamma","alpha","none"])
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--zeta_norm", type=str, default="mean", choices=["mean","none","p95"])
    ap.add_argument("--out_json", type=str, default=None)

    # 그리드 모드
    ap.add_argument("--grid", action="store_true", help="그리드 탐색 모드 on")
    ap.add_argument("--gammas", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0", help="쉼표구분 리스트")
    ap.add_argument("--zeta_norms", type=str, default="mean,p95,none", help="쉼표구분 리스트")
    ap.add_argument("--blend_modes", type=str, default="gamma,alpha", help="쉼표구분 리스트")
    ap.add_argument("--out_dir", type=str, default=None, help="그리드 결과 저장 폴더(지정 없으면 dml_root/grid_waae)")

    args = ap.parse_args()

    bonus_dir = Path(args.bonus_dir)
    dml_root  = Path(args.dml_root)
    ckpt      = Path(args.ckpt)
    shift_dir = Path(args.shift_outputs_dir) if args.shift_outputs_dir else None

    if not args.grid:
        # 기존 단일 실행 (호환 유지)
        out_json = Path(args.out_json) if args.out_json else (dml_root / "07_waae_multi_report.json")
        run_setting(
            bonus_dir=bonus_dir, dml_root=dml_root, ckpt=ckpt,
            shift_outputs_dir=shift_dir, waae_mode=args.waae_mode,
            blend=args.blend, gamma=args.gamma, zeta_norm=args.zeta_norm,
            out_json=out_json
        )
        return

    # --- 그리드 모드 ---
    out_dir = Path(args.out_dir) if args.out_dir else (dml_root / "grid_waae")
    out_dir.mkdir(parents=True, exist_ok=True)

    gammas = parse_list_floats(args.gammas)
    zeta_norms = parse_list_str(args.zeta_norms)
    blends = parse_list_str(args.blend_modes)

    summary = []
    for blend in blends:
        if blend == "gamma":
            for zn in zeta_norms:
                for g in gammas:
                    fname = f"07_multi_{args.waae_mode}_blend-gamma_zeta-{zn}_gamma-{g}.json"
                    r = run_setting(
                        bonus_dir=bonus_dir, dml_root=dml_root, ckpt=ckpt,
                        shift_outputs_dir=shift_dir, waae_mode=args.waae_mode,
                        blend="gamma", gamma=g, zeta_norm=zn,
                        out_json=out_dir / fname
                    )
                    summary.append({"blend": "gamma", "zeta_norm": zn, "gamma": g, **r})
        elif blend == "alpha":
            for zn in zeta_norms:
                fname = f"07_multi_{args.waae_mode}_blend-alpha_zeta-{zn}.json"
                r = run_setting(
                    bonus_dir=bonus_dir, dml_root=dml_root, ckpt=ckpt,
                    shift_outputs_dir=shift_dir, waae_mode=args.waae_mode,
                    blend="alpha", gamma=0.0, zeta_norm=zn,
                    out_json=out_dir / fname
                )
                summary.append({"blend": "alpha", "zeta_norm": zn, "gamma": None, **r})
        elif blend == "none":
            fname = f"07_multi_{args.waae_mode}_blend-none.json"
            r = run_setting(
                bonus_dir=bonus_dir, dml_root=dml_root, ckpt=ckpt,
                shift_outputs_dir=shift_dir, waae_mode=args.waae_mode,
                blend="none", gamma=0.0, zeta_norm="none",
                out_json=out_dir / fname
            )
            summary.append({"blend": "none", "zeta_norm": "none", "gamma": None, **r})

    # 마스터 요약 저장 + 콘솔 테이블
    master = sorted(summary, key=lambda s: s["relative_improvement_mean"], reverse=True)
    master_path = out_dir / "07_grid_summary.json"
    with open(master_path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"\n[MASTER] saved -> {master_path}")
    print(f"{'rank':<4} {'blend':<6} {'zeta':<6} {'gamma':<5} {'RelImpr%':>9} {'PFN_mean':>10} {'DML_mean':>10}")
    for i, s in enumerate(master, 1):
        gi = "" if s["gamma"] is None else str(s["gamma"])
        print(f"{i:<4} {s['blend']:<6} {s['zeta_norm']:<6} {gi:<5} "
              f"{100*s['relative_improvement_mean']:>8.2f}% {s['PFN_mean']:>10.6f} {s['DML_mean']:>10.6f}")

if __name__ == "__main__":
    main()


# import os, json, argparse, glob
# from pathlib import Path
# import numpy as np
# import torch
#
# def load_bonus_arrays(bonus_dir: Path):
#     X = np.load(bonus_dir / "X.npy").astype(np.float32)
#     T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
#     Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
#     return X, T, Y
#
# def waae_kpi(Y, T, mu0_pred, mu1_pred):
#     m0 = (T==0); m1 = (T==1)
#     r0 = float(m0.mean()); r1 = 1.0 - r0
#     y0 = float(Y[m0].mean()) if m0.any() else 0.0
#     y1 = float(Y[m1].mean()) if m1.any() else 0.0
#     e0 = abs(float(mu0_pred[m0].mean()) - y0) if m0.any() else 0.0
#     e1 = abs(float(mu1_pred[m1].mean()) - y1) if m1.any() else 0.0
#     return r0*e0 + r1*e1
#
# def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_anchor, mu1_anchor):
#     r1 = float((T==1).mean()); r0 = 1.0 - r1
#     mu0_bar_hat = float(mu0_pred.mean()); mu1_bar_hat = float(mu1_pred.mean())
#     e1 = np.clip(e_hat, 1e-3, 1-1e-3); e0 = 1.0 - e1
#     ind1 = (T==1).astype(np.float32); ind0 = 1.0 - ind1
#     # AIPW 평균 (anchor는 DML OOF 추정치 역할)
#     aipw0 = mu0_anchor + ind0/e0 * (Y - mu0_anchor)
#     aipw1 = mu1_anchor + ind1/e1 * (Y - mu1_anchor)
#     mu0_bar = float(aipw0.mean()); mu1_bar = float(aipw1.mean())
#     err0 = abs(mu0_bar_hat - mu0_bar); err1 = abs(mu1_bar_hat - mu1_bar)
#     return r0*err0 + r1*err1
#
# def load_tau_from_ckpt(X, ckpt_path, device="cpu"):
#     n = len(X)
#     # try npz first
#     try:
#         with np.load(ckpt_path) as z:
#             if "tau_pfn" in z:
#                 tau = z["tau_pfn"].astype(np.float32).reshape(-1)
#                 assert len(tau) == n
#                 return tau
#     except Exception:
#         pass
#     # torch
#     ckpt = torch.load(ckpt_path, map_location=device)
#     tau = ckpt.get("tau_pfn", None)
#     if tau is None:
#         raise KeyError("ckpt has no 'tau_pfn'")
#     if torch.is_tensor(tau):
#         tau = tau.detach().cpu().numpy()
#     tau = tau.astype(np.float32).reshape(-1)
#     assert len(tau) == n
#     return tau
#
# def eval_one_seed(bonus_dir: Path, dml_dir: Path, ckpt: Path,
#                   waae_mode: str, blend: str, gamma: float,
#                   zeta_path: Path|None, zeta_norm: str):
#     X, T, Y = load_bonus_arrays(bonus_dir)
#     mu0 = np.load(dml_dir / "mu0_hat.npy")
#     mu1 = np.load(dml_dir / "mu1_hat.npy")
#     e   = np.load(dml_dir / "e_hat.npy")
#
#     # PFN(+DML)
#     tau_pfn = load_tau_from_ckpt(X, ckpt)
#
#     # -------------------- [추가] orientation check --------------------
#     # DML의 T=1 잔차와 PFN tau의 상관이 음수면 tau 부호를 뒤집어 정렬
#     mask1 = (T == 1)
#     res1  = (Y - mu1)[mask1]                    # DML μ1 잔차
#     tau1  = tau_pfn[mask1]
#     eps   = 1e-8
#     corr  = float(np.corrcoef(res1, tau1)[0, 1]) if (res1.std() > eps and tau1.std() > eps) else 0.0
#     if corr < 0.0:
#         tau_pfn = -tau_pfn
#     # -----------------------------------------------------------------
#
#     mu0_pfn = mu0.copy()
#     mu1_raw = mu0 + tau_pfn
#     mu1_pfn = mu1_raw.copy()
#
#     # load zeta if exists
#     zeta = None; z = None
#     if zeta_path is not None and zeta_path.exists():
#         zeta = np.load(str(zeta_path)).astype(np.float32)
#         if zeta_norm == "mean":
#             z = zeta / max(float(zeta.mean()), 1e-8)
#         elif zeta_norm == "p95":
#             q = float(np.quantile(zeta, 0.95))
#             z = zeta / max(q, 1e-8)
#         else:  # none
#             z = zeta.copy()
#
#     # blend
#     if blend == "none":
#         mu1_pfn = mu1_raw
#     elif blend == "gamma":
#         if z is None:
#             w = np.full_like(mu1, gamma, dtype=np.float32)
#         else:
#             w = np.clip(gamma * z, 0.0, 1.0).astype(np.float32)
#         mu1_pfn = (1.0 - w) * mu1 + w * mu1_raw
#     else:  # alpha mean-preserving
#         if z is None:
#             tau_w_mean = float(tau_pfn.mean())
#         else:
#             tau_w_mean = float((z * tau_pfn).mean())
#         if waae_mode == "paper":
#             e1 = np.clip(e, 1e-3, 1-1e-3); ind1 = (T==1).astype(np.float32)
#             aipw1 = mu1 + ind1/e1 * (Y - mu1)
#             m1_true = float(aipw1.mean())
#         else:
#             mask1 = (T==1)
#             m1_true = float(Y[mask1].mean()) if mask1.any() else float(Y.mean())
#         m1_dml = float(mu1.mean())
#         alpha = 0.0 if abs(tau_w_mean) < 1e-10 else (m1_true - m1_dml) / tau_w_mean
#         mu1_pfn = mu1 + alpha * (tau_pfn if z is None else (z * tau_pfn))
#
#     # metrics
#     if waae_mode == "paper":
#         dml_waae = waae_paper(Y, T, mu0, mu1, e, mu0, mu1)
#         pfn_waae = waae_paper(Y, T, mu0_pfn, mu1_pfn, e, mu0, mu1)
#     else:
#         dml_waae = waae_kpi(Y, T, mu0, mu1)
#         pfn_waae = waae_kpi(Y, T, mu0_pfn, mu1_pfn)
#
#     return dict(DML=float(dml_waae), PFN_plus_DML=float(pfn_waae))
#
# def main():
#     ap = argparse.ArgumentParser("Evaluate WAAE across multiple seed_* dirs and aggregate")
#     ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus")
#     ap.add_argument("--dml_root", type=str, required=True,
#                     help="05 결과 루트 (seed_* 하위에 mu0/mu1/e_hat.npy 존재)")
#     ap.add_argument("--ckpt", type=str, required=True,
#                     help="06_pfnlite.ckpt (tau_pfn 포함)")
#     ap.add_argument("--shift_outputs_dir", type=str, default=None,
#                     help="zeta 파일(05_zeta_bonus.npy) 위치. 생략 시 미사용(=no-shift).")
#     ap.add_argument("--waae_mode", type=str, default="paper", choices=["paper","kpi"])
#     ap.add_argument("--blend", type=str, default="gamma", choices=["gamma","alpha","none"])
#     ap.add_argument("--gamma", type=float, default=0.8)
#     ap.add_argument("--zeta_norm", type=str, default="mean", choices=["mean","none","p95"])
#     ap.add_argument("--out_json", type=str, default=None)
#     args = ap.parse_args()
#
#     bonus_dir = Path(args.bonus_dir)
#     dml_root  = Path(args.dml_root)
#     ckpt      = Path(args.ckpt)
#     zeta_path = None
#     if args.shift_outputs_dir:
#         zeta_path = Path(args.shift_outputs_dir) / "05_zeta_bonus.npy"
#
#     # seed_dirs = sorted([Path(p) for p in glob.glob(str(dml_root / "seed_*")) if Path(p).is_dir()])
#     # if not seed_dirs:
#     #     raise FileNotFoundError(f"No seed_* dirs under {dml_root}")
#     if dml_root.name.startswith("seed_") and dml_root.is_dir():
#         seed_dirs = [dml_root]
#     else:
#         seed_dirs = sorted([Path(p) for p in glob.glob(str(dml_root / "seed_*")) if Path(p).is_dir()])
#         if not seed_dirs:
#             raise FileNotFoundError(f"No seed_* dirs under {dml_root}")
#
#     rows = []
#     for sd in seed_dirs:
#         r = eval_one_seed(
#             bonus_dir=bonus_dir, dml_dir=sd, ckpt=ckpt,
#             waae_mode=args.waae_mode, blend=args.blend, gamma=args.gamma,
#             zeta_path=zeta_path, zeta_norm=args.zeta_norm
#         )
#         rows.append({"seed_dir": sd.name, **r})
#         print(f"[{sd.name}] DML={r['DML']:.6f} | PFN+DML={r['PFN_plus_DML']:.6f}")
#
#     # (07_eval_waae_multi.py) 집계 후 출력 부분만 교체
#
#     dml_list = [x["DML"] for x in rows]
#     pfn_list = [x["PFN_plus_DML"] for x in rows]
#     imprs = [(d - p) / d if d > 0 else 0.0 for d, p in zip(dml_list, pfn_list)]
#     n_better = sum(1 for d, p in zip(dml_list, pfn_list) if p < d)
#
#     agg = {
#         "DML_mean": float(np.mean(dml_list)),
#         "DML_std": float(np.std(dml_list, ddof=1)) if len(dml_list) > 1 else 0.0,
#         "PFN_plus_DML_mean": float(np.mean(pfn_list)),
#         "PFN_plus_DML_std": float(np.std(pfn_list, ddof=1)) if len(pfn_list) > 1 else 0.0,
#         "relative_improvement_mean": float(np.mean(imprs)),
#         "n_runs": len(rows),
#         "n_runs_better": int(n_better)
#     }
#
#     print(f"[SUMMARY] "
#           f"DML mean={agg['DML_mean']:.6f} (±{agg['DML_std']:.6f}), "
#           f"PFN+DML mean={agg['PFN_plus_DML_mean']:.6f} (±{agg['PFN_plus_DML_std']:.6f}), "
#           f"Rel.Impr mean={agg['relative_improvement_mean']:.3%}, "
#           f"better_seeds={agg['n_runs_better']}/{agg['n_runs']}")
#
#     # dml_list  = [x["DML"] for x in rows]
#     # pfn_list  = [x["PFN_plus_DML"] for x in rows]
#     # agg = {
#     #     "DML_mean": float(np.mean(dml_list)),
#     #     "DML_std":  float(np.std(dml_list, ddof=1)) if len(dml_list)>1 else 0.0,
#     #     "PFN_mean": float(np.mean(pfn_list)),
#     #     "PFN_std":  float(np.std(pfn_list, ddof=1)) if len(pfn_list)>1 else 0.0,
#     #     "relative_improvement_mean": float(np.mean([(d-p)/d if d>0 else 0.0 for d,p in zip(dml_list,pfn_list)])),
#     #     "n_runs": len(rows)
#     # }
#     # report = {
#     #     "mode": args.waae_mode,
#     #     "blend": args.blend,
#     #     "gamma": float(args.gamma),
#     #     "zeta_norm": args.zeta_norm,
#     #     "rows": rows,
#     #     "aggregate": agg,
#     #     "paths": {
#     #         "dml_root": str(dml_root),
#     #         "ckpt": str(ckpt),
#     #         "zeta_path": str(zeta_path) if zeta_path is not None else None
#     #     }
#     # }
#     # out_json = Path(args.out_json) if args.out_json else (dml_root / "07_waae_multi_report.json")
#     # with open(out_json, "w") as f:
#     #     json.dump(report, f, indent=2)
#     # print(f"[OK] saved -> {out_json}")
#     # print(f"[SUMMARY] DML mean={agg['DML_mean']:.6f} (±{agg['DML_std']:.6f}), "
#     #       f"PFN mean={agg['PFN_mean']:.6f} (±{agg['PFN_std']:.6f}), "
#     #       f"Rel.Impr mean={agg['relative_improvement_mean']:.3%}")
#
# if __name__ == "__main__":
#     main()
