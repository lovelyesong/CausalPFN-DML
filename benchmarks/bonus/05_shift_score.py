# benchmarks/bonus/05_shift_score.py

### Termina에 아래와 같이 zeta mean 확인 가능
# python - <<'PY'
# import numpy as np
# z = np.load("benchmarks/bonus/outputs/05_zeta_bonus.npy")
# print("mean(zeta)=", z.mean())
# PY


import os, glob, json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------- helpers ----------
def find_repo_root(start: Path) -> Path:
    # Walk upwards to find the repo root by locating a .git directory.
    # Keeps paths robust regardless of the current working directory.
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    # Fallback: climb a few parents if .git is not found (last resort).
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]

def iter_prior_X(prior_dir, max_rows=100_000, seed=42):
    """
    Stream only X from prior shards (shard_*.npz) and subsample up to max_rows.
    This keeps memory usage low and speeds things up for large prior pools.
    """
    paths = sorted(glob.glob(os.path.join(prior_dir, "shard_*.npz")))
    rng = np.random.default_rng(seed)
    took = 0
    for p in paths:
        d = np.load(p)
        X = d["X"]  # use only features from each shard
        if took + len(X) <= max_rows:
            yield X
            took += len(X)
        else:
            # Take just enough rows from the last shard to reach max_rows.
            need = max_rows - took
            if need > 0:
                idx = rng.integers(0, len(X), size=need)
                yield X[idx]
                took += need
            break

# ---------- main ----------
def main():
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    # Directories: prior shards (.npz), Bonus raw data, and outputs
    PRIOR_DIR = (repo_root / "data/prior/bonus").resolve()
    BONUS_DIR = (repo_root / "notebooks/bonus_benchmarks/data/bonus").resolve()
    OUT_DIR   = (repo_root / "benchmarks/bonus/outputs").resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Bonus X (domain=1)
    X_bonus = np.load(BONUS_DIR / "X.npy").astype(np.float32)
    n_bonus, p = X_bonus.shape
    print(f"[info] bonus X: {X_bonus.shape}")

    # Collect prior X from shards (domain=0), capping rows for speed.
    X_prior = np.concatenate(list(iter_prior_X(PRIOR_DIR, max_rows=100_000)), axis=0).astype(np.float32)
    n_prior = X_prior.shape[0]
    print(f"[info] prior  X: {X_prior.shape} (from {PRIOR_DIR})")

    # Build domain-classification dataset: prior=0, bonus=1
    X_all = np.vstack([X_prior, X_bonus])
    y_all = np.hstack([np.zeros(n_prior, dtype=int), np.ones(n_bonus, dtype=int)])

    # Buffer for out-of-fold (OOF) probabilities
    oof = np.zeros(len(y_all), dtype=float)
    # Stratified 5-fold CV to keep class balance and reduce optimism bias
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

    # Simple, stable, reasonably calibrated probability model
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logit", LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000, n_jobs=None))
    ])

    # Fill OOF probabilities: oof = P(bonus | x)
    for k, (tr, te) in enumerate(skf.split(X_all, y_all), 1):
        clf.fit(X_all[tr], y_all[tr])
        oof[te] = clf.predict_proba(X_all[te])[:, 1]  # probability of class 1 (bonus)
        print(f"[cv {k}/5] done.")

    # Diagnostic: domain separability. Higher AUC ⇒ larger prior↔bonus shift
    auc = roc_auc_score(y_all, oof)
    print(f"[metric] domain AUC (higher = more shift): {auc:.4f}")

    # Extract Bonus-side probabilities in original order:
    # zeta(x) = P(bonus | x) for each Bonus sample
    zeta_bonus = oof[n_prior:]  # length = n_bonus
    print(f"[diag] zeta_bonus mean={float(zeta_bonus.mean()):.4f}  "
          f"min={float(zeta_bonus.min()):.4f}  max={float(zeta_bonus.max()):.4f}")

    # Save artifacts: per-sample zeta and metadata
    np.save(OUT_DIR / "05_zeta_bonus.npy", zeta_bonus.astype(np.float32))
    meta = {
        "prior_dir": str(PRIOR_DIR),
        "bonus_dir": str(BONUS_DIR),
        "n_prior": int(n_prior),
        "n_bonus": int(n_bonus),
        "p": int(p),
        "cv": 5,
        "clf": "StandardScaler + LogisticRegression",
        "auc": float(auc),
        # Usage hint: use w=zeta or w=(1-zeta) as weights for PFN loss/ensemble.
        "note": "zeta = P(bonus|x) out-of-fold prob per sample; use e.g. w=zeta or w=(1-zeta) as weights."
    }
    with open(OUT_DIR / "05_shift_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] saved -> {OUT_DIR / '05_zeta_bonus.npy'} and 05_shift_meta.json")

if __name__ == "__main__":
    main()
