# benchmarks/bonus/03_doubleml_baseline.py
import os, json
import numpy as np
from pathlib import Path
from doubleml import DoubleMLData, DoubleMLIRM, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import time, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]

def main():
    # --- Paths ---
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    bonus_dir = (repo_root / "notebooks/bonus_benchmarks/data/bonus").resolve()
    out_dir   = (repo_root / "benchmarks/bonus/outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] repo_root={repo_root}")
    print(f"[info] BONUS_DIR={bonus_dir}")

    # --- Load arrays ---
    X = np.load(bonus_dir / "X.npy")
    T = np.load(bonus_dir / "T.npy").reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").reshape(-1)
    print(f"[info] X={X.shape}, T={T.shape}, Y={Y.shape}")

    # --- DoubleML: IRM with score='ATE' ---
    dml_data = DoubleMLData.from_arrays(x=X, y=Y, d=T)

    ml_g = RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42, n_jobs=-1)  # E[Y|T,X]
    ml_m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=43, n_jobs=-1)  # P[T=1|X]

    dml = DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m,n_folds=5, score="ATE", draw_sample_splitting=True) #IRM(Interactive Regression Model)
    # dml = DoubleMLPLR(dml_data, ml_l=ml_g, ml_m=ml_m, n_folds=5, score="partialling out") #DoubleMLPLR


    t0 = time.time()
    print("[info] Fitting DML (IRM)…")
    dml.fit()
    print(f"[timing] fit done in {time.time() - t0:.1f}s")

    ate = float(dml.coef)
    se  = float(dml.se)
    ci  = (float(dml.confint().iloc[0, 0]), float(dml.confint().iloc[0, 1]))

    print("\n[DML Baseline via IRM]")
    # print("\n[DML Baseline via PLR]")
    print(f"ATE    : {ate:.6f}")
    print(f"SE     : {se:.6f}")
    print(f"95% CI : [{ci[0]:.6f}, {ci[1]:.6f}]")

    with open(out_dir / "03_dml_ate.json", "w") as f:
        json.dump({"ATE": ate, "SE": se, "CI95": [ci[0], ci[1]]}, f, indent=2)
    print(f"[OK] saved -> {out_dir / '03_dml_ate.json'}")

if __name__ == "__main__":
    main()
