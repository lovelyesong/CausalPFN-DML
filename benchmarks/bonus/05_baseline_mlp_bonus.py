# benchmarks/bonus/05_baseline_mlp_bonus.py

# python benchmarks/bonus/05_baseline_mlp_bonus.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --outputs_dir benchmarks/bonus/outputs_mlp_baseline \
#   --waae_mode paper \
#   --reps 20 --kfold 5 \
#   --epochs 50 --batch_size 512 --lr 1e-3 --hidden 256 --depth 3


# python benchmarks/bonus/05_baseline_mlp_bonus.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --outputs_dir benchmarks/bonus/outputs_mlp_baseline_kpi \
#   --waae_mode kpi --reps 20 --kfold 5

# python benchmarks/bonus/05_baseline_mlp_bonus.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --outputs_dir benchmarks/bonus/outputs_mlp05 \
#   --waae_mode paper \
#   --reps 20 --kfold 5 \
#   --epochs 50 --batch_size 512 --lr 1e-3



import os, json, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

# ---------------- I/O ----------------
def load_bonus_arrays(bonus_dir: Path):
    X = np.load(bonus_dir / "X.npy").astype(np.float32)
    T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
    return X, T, Y

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------- Models ---------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden=256, depth=3, dropout=0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ------------- Training utils -------------
def torch_fit_regressor(Xtr, ytr, Xte, *, hidden, depth, dropout, lr, epochs, bs, device):
    model = MLP(Xtr.shape[1], 1, hidden, depth, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(Xtr)
    xb = torch.tensor(Xtr, device=device)
    yb = torch.tensor(ytr, device=device).unsqueeze(1)
    model.train()
    for _ in range(epochs):
        # mini-batch
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            j = idx[i:i+bs]
            pred = model(xb[j])
            loss = loss_fn(pred, yb[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xte, device=device)).squeeze(1).cpu().numpy().astype(np.float32)
    return out

def torch_fit_classifier(Xtr, ttr, Xte, *, hidden, depth, dropout, lr, epochs, bs, device):
    model = MLP(Xtr.shape[1], 1, hidden, depth, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    xb = torch.tensor(Xtr, device=device)
    tb = torch.tensor(ttr, device=device).float().unsqueeze(1)
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            j = idx[i:i+bs]
            logit = model(xb[j])
            loss = loss_fn(logit, tb[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(Xte, device=device)).squeeze(1)
        prob = torch.sigmoid(logit).cpu().numpy().astype(np.float32)
    return prob

# ------------- Metrics (r69 WAAE) -------------
def waae_kpi(Y, T, mu0_pred, mu1_pred):
    m0, m1 = (T==0), (T==1)
    r0 = float(m0.mean()); r1 = 1.0 - r0
    y0 = float(Y[m0].mean()) if m0.any() else 0.0
    y1 = float(Y[m1].mean()) if m1.any() else 0.0
    e0 = abs(float(mu0_pred[m0].mean()) - y0) if m0.any() else 0.0
    e1 = abs(float(mu1_pred[m1].mean()) - y1) if m1.any() else 0.0
    return r0*e0 + r1*e1

def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_hat_dml=None, mu1_hat_dml=None):
    # 여기서는 baseline MLP가 곧 "hat" 역할을 하므로 mu*_hat_dml 인자 없이 자체로 DR 평균을 구성
    # r69 정의: WAAE(\hat μ) = Σ_t | mean(\hat μ_t) - mean(μ_t) | * P_N(t)
    # DR 평균(μ_t)은 AIPW로 계산
    n = len(Y)
    r1 = float((T==1).mean()); r0 = 1.0 - r1
    mu0_bar_hat = float(mu0_pred.mean()); mu1_bar_hat = float(mu1_pred.mean())

    e1 = np.clip(e_hat, 1e-3, 1-1e-3); e0 = 1.0 - e1
    ind1 = (T==1).astype(np.float32); ind0 = 1.0 - ind1

    # AIPW 평균
    aipw0 = mu0_pred + ind0/e0 * (Y - mu0_pred)
    aipw1 = mu1_pred + ind1/e1 * (Y - mu1_pred)
    mu0_bar = float(aipw0.mean()); mu1_bar = float(aipw1.mean())

    err0 = abs(mu0_bar_hat - mu0_bar); err1 = abs(mu1_bar_hat - mu1_bar)
    return r0*err0 + r1*err1

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser("BONUS baseline MLP with repeated runs + WAAE aggregation")
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus")
    ap.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs")
    ap.add_argument("--waae_mode", type=str, default="paper", choices=["paper","kpi"])
    ap.add_argument("--reps", type=int, default=10, help="number of repeated runs (different seeds)")
    ap.add_argument("--kfold", type=int, default=5, help="OOF folds")
    ap.add_argument("--base_seed", type=int, default=2025)

    # model hparams
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bonus_dir = Path(args.bonus_dir); out_dir = Path(args.outputs_dir)
    ensure_dir(out_dir)

    X, T, Y = load_bonus_arrays(bonus_dir)
    n, p = X.shape
    kf = StratifiedKFold(n_splits=args.kfold, shuffle=True)

    waae_runs = []
    mu0_runs, mu1_runs, e_runs = [], [], []

    for r in range(args.reps):
        seed = args.base_seed + r
        np.random.seed(seed); torch.manual_seed(seed)
        mu0_oof = np.zeros(n, dtype=np.float32)
        mu1_oof = np.zeros(n, dtype=np.float32)
        e_oof   = np.zeros(n, dtype=np.float32)

        for tr_idx, te_idx in kf.split(X, T):
            Xtr, Xte = X[tr_idx], X[te_idx]
            Ttr, Tte = T[tr_idx], T[te_idx]
            Ytr, Yte = Y[tr_idx], Y[te_idx]

            # fit mu0 on T=0
            m0 = (Ttr==0)
            mu0_te = torch_fit_regressor(
                Xtr[m0], Ytr[m0], Xte,
                hidden=args.hidden, depth=args.depth, dropout=args.dropout,
                lr=args.lr, epochs=args.epochs, bs=args.batch_size, device=device
            )
            # fit mu1 on T=1
            m1 = (Ttr==1)
            mu1_te = torch_fit_regressor(
                Xtr[m1], Ytr[m1], Xte,
                hidden=args.hidden, depth=args.depth, dropout=args.dropout,
                lr=args.lr, epochs=args.epochs, bs=args.batch_size, device=device
            )
            # fit e(x)
            e_te = torch_fit_classifier(
                Xtr, Ttr, Xte,
                hidden=args.hidden, depth=args.depth, dropout=args.dropout,
                lr=args.lr, epochs=args.epochs, bs=args.batch_size, device=device
            )

            mu0_oof[te_idx] = mu0_te
            mu1_oof[te_idx] = mu1_te
            e_oof[te_idx]   = e_te

        # compute WAAE for this run
        if args.waae_mode == "paper":
            waae = waae_paper(Y, T, mu0_oof, mu1_oof, e_oof)
        else:
            waae = waae_kpi(Y, T, mu0_oof, mu1_oof)

        waae_runs.append(float(waae))
        mu0_runs.append(mu0_oof); mu1_runs.append(mu1_oof); e_runs.append(e_oof)

        # save per-seed artifacts (for 디버깅/재현)
        seed_dir = out_dir / f"seed_{seed}"
        ensure_dir(seed_dir)
        np.save(seed_dir / "mu0_hat.npy", mu0_oof)
        np.save(seed_dir / "mu1_hat.npy", mu1_oof)
        np.save(seed_dir / "e_hat.npy",   e_oof)

        print(f"[rep {r+1}/{args.reps}] seed={seed} WAAE={waae:.6f}")

    # aggregate (mean over runs)
    mu0_mean = np.mean(np.stack(mu0_runs, axis=0), axis=0).astype(np.float32)
    mu1_mean = np.mean(np.stack(mu1_runs, axis=0), axis=0).astype(np.float32)
    e_mean   = np.mean(np.stack(e_runs,   axis=0), axis=0).astype(np.float32)

    # save “average” artifacts for downstream steps (06/07가 기대하는 경로)
    np.save(out_dir / "mu0_hat.npy", mu0_mean)
    np.save(out_dir / "mu1_hat.npy", mu1_mean)
    np.save(out_dir / "e_hat.npy",   e_mean)

    # recompute WAAE on the averaged predictions (참고용)
    if args.waae_mode == "paper":
        waae_avg_pred = waae_paper(Y, T, mu0_mean, mu1_mean, e_mean)
    else:
        waae_avg_pred = waae_kpi(Y, T, mu0_mean, mu1_mean)

    rep_mean = float(np.mean(waae_runs))
    rep_std  = float(np.std(waae_runs, ddof=1)) if len(waae_runs) > 1 else 0.0

    report = {
        "mode": args.waae_mode,
        "reps": int(args.reps),
        "kfold": int(args.kfold),
        "waae_per_run": waae_runs,
        "waae_mean_over_runs": rep_mean,
        "waae_std_over_runs": rep_std,
        "waae_on_averaged_predictions": float(waae_avg_pred),
        "paths": {
            "outputs_dir": str(out_dir),
            "avg_mu_files": {
                "mu0_hat": str(out_dir / "mu0_hat.npy"),
                "mu1_hat": str(out_dir / "mu1_hat.npy"),
                "e_hat":   str(out_dir / "e_hat.npy"),
            }
        },
        "note": "WAAE per r69 uses DR/AIPW means in 'paper' mode; values aggregated across repeated runs."
    }
    out_json = Path(args.out_json) if args.out_json else (out_dir / "05_baseline_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] saved report -> {out_json}")
    print(f"[SUMMARY] WAAE mean={rep_mean:.6f} (std={rep_std:.6f}); WAAE(avg-preds)={waae_avg_pred:.6f}")

if __name__ == "__main__":
    main()
