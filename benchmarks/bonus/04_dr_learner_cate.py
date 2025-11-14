# benchmarks/bonus/04_dr_learner_cate.py
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]

def main():
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    bonus_dir = (repo_root / "notebooks/bonus_benchmarks/data/bonus").resolve()
    out_dir   = (repo_root / "benchmarks/bonus/outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(bonus_dir / "X.npy")
    T = np.load(bonus_dir / "T.npy").reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").reshape(-1)
    n, p = X.shape
    print(f"[info] X={X.shape}, T={T.shape}, Y={Y.shape}")

    def new_g(): return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=1, n_jobs=-1)
    def new_m(): return RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=2, n_jobs=-1)
    def new_t(): return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=3, n_jobs=-1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mu0_hat = np.zeros(n, dtype=float)
    mu1_hat = np.zeros(n, dtype=float)
    e_hat   = np.zeros(n, dtype=float)
    psi     = np.zeros(n, dtype=float)  # DR pseudo-outcome

    for fold, (tr, te) in enumerate(kf.split(X), 1):
        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]
        Ttr, Tte = T[tr], T[te]

        g0, g1 = new_g(), new_g()
        g0.fit(Xtr[Ttr == 0], Ytr[Ttr == 0])
        g1.fit(Xtr[Ttr == 1], Ytr[Ttr == 1])

        m = new_m()
        m.fit(Xtr, Ttr)

        mu0_te = g0.predict(Xte)
        mu1_te = g1.predict(Xte)
        e_te   = m.predict_proba(Xte)[:, 1]
        e_te   = np.clip(e_te, 1e-3, 1-1e-3)

        mu0_hat[te] = mu0_te
        mu1_hat[te] = mu1_te
        e_hat[te]   = e_te

        mu_te = np.where(Tte == 1, mu1_te, mu0_te)
        numer = (Tte - e_te) * (Yte - mu_te)
        denom = e_te * (1 - e_te)
        psi[te] = numer / denom + (mu1_te - mu0_te)

        print(f"[fold {fold}/5] done")

    tau_learner = new_t()
    tau_learner.fit(X, psi)
    tau_hat = tau_learner.predict(X).astype(np.float32)

    np.save(out_dir / "tau_hat.npy", tau_hat)
    np.save(out_dir / "e_hat.npy",   e_hat.astype(np.float32))
    np.save(out_dir / "mu0_hat.npy", mu0_hat.astype(np.float32))
    np.save(out_dir / "mu1_hat.npy", mu1_hat.astype(np.float32))

    summary = {
        "tau_hat_abs_mean": float(np.mean(np.abs(tau_hat))),
        "e_hat_mean": float(e_hat.mean()),
        "n": int(n), "p": int(p)
    }
    with open(out_dir / "04_dr_learner_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[OK] saved tau_hat/e_hat/mu0_hat/mu1_hat and summary")

if __name__ == "__main__":
    main()
