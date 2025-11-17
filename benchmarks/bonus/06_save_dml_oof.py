import argparse, json
from pathlib import Path
import numpy as np

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# python benchmarks/bonus/04_save_dml_oof.py \
#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --outputs_dir benchmarks/bonus/outputs

# Reproducibility
SEED = 2025

def load_bonus_arrays(bonus_dir: Path):
    X = np.load(bonus_dir / "X.npy").astype(np.float32)
    T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
    return X, T, Y

def main():
    ap = argparse.ArgumentParser(description="Save DML-style OOF predictions: mu0_hat.npy, mu1_hat.npy, e_hat.npy.")
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus",
                    help="Folder with X.npy/T.npy/Y.npy.")
    ap.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs",
                    help="Where to write mu0_hat.npy, mu1_hat.npy, e_hat.npy and a meta.json.")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    bonus_dir = Path(args.bonus_dir).resolve()
    out_dir = Path(args.outputs_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X, T, Y = load_bonus_arrays(bonus_dir)
    n, p = X.shape
    print(f"[info] X={X.shape}, T={T.shape}, Y={Y.shape}")

    # Define nuisance learners (simple, deterministic)
    # mu_t(x): regression; e(x): classification
    mu_learner = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64),
                             activation="relu", random_state=SEED,
                             max_iter=300, learning_rate_init=1e-3))
    ])
    e_learner = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64,),
                              activation="relu", random_state=SEED,
                              max_iter=300, learning_rate_init=1e-3))
    ])

    # OOF containers
    mu0_hat = np.zeros(n, dtype=np.float32)
    mu1_hat = np.zeros(n, dtype=np.float32)
    e_hat   = np.zeros(n, dtype=np.float32)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    for k, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X[tr], X[va]
        T_tr, T_va = T[tr], T[va]
        Y_tr, Y_va = Y[tr], Y[va]

        # Propensity e(x) using classifier on full train fold
        e_model = e_learner
        e_model.fit(X_tr, T_tr)
        e_hat_va = e_model.predict_proba(X_va)[:, 1]
        e_hat[va] = e_hat_va.astype(np.float32)

        # mu0(x): fit on control group of train fold
        mu0_model = mu_learner
        mask0 = (T_tr == 0)
        mu0_model.fit(X_tr[mask0], Y_tr[mask0])
        mu0_hat[va] = mu0_model.predict(X_va).astype(np.float32)

        # mu1(x): fit on treated group of train fold
        mu1_model = mu_learner
        mask1 = (T_tr == 1)
        mu1_model.fit(X_tr[mask1], Y_tr[mask1])
        mu1_hat[va] = mu1_model.predict(X_va).astype(np.float32)

        print(f"[fold {k}] |va|={len(va)}  e_mean={e_hat_va.mean():.3f}  "
              f"mu0_mean={mu0_hat[va].mean():.3f}  mu1_mean={mu1_hat[va].mean():.3f}")

    # Persist
    np.save(out_dir / "mu0_hat.npy", mu0_hat)
    np.save(out_dir / "mu1_hat.npy", mu1_hat)
    np.save(out_dir / "e_hat.npy",   e_hat)

    meta = dict(seed=SEED, n_folds=args.n_folds,
                mu_model="MLPRegressor(128,64)", e_model="MLPClassifier(64)",
                bonus_dir=str(bonus_dir), outputs_dir=str(out_dir))
    (out_dir / "meta_dml.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] saved to {out_dir}")

if __name__ == "__main__":
    main()
