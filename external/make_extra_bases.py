#!/usr/bin/env python3
"""
Create extra base tables for prior construction.

- Option A: fetch from OpenML via scikit-learn (e.g., 'credit-g', 'adult')
- Option B: convert local CSVs into numeric matrices
- Optional: match feature dimension p to Bonus X via PCA or random projection

Usage examples
--------------
# Make both targets in your paths (OpenML fetch; save numeric .npy and .csv)
python scripts/make_extra_bases.py \
  --bonus_x /home/yesong/CausalPFN-DML/notebooks/bonus_benchmarks/data/bonus/X.npy \
  --save_npy /home/yesong/CausalPFN-DML/external/openml/credit_X.npy \
  --save_csv /home/yesong/CausalPFN-DML/external/uci/census_X.csv \
  --openml credit-g adult --match_mode pca

# Convert any local CSV to numeric npy/csv (no internet needed)
python scripts/make_extra_bases.py \
  --bonus_x /home/yesong/CausalPFN-DML/notebooks/bonus_benchmarks/data/bonus/X.npy \
  --local_csv /path/to/your_numeric_only.csv \
  --save_npy /home/yesong/CausalPFN-DML/external/openml/credit_X.npy \
  --save_csv /home/yesong/CausalPFN-DML/external/uci/census_X.csv \
  --match_mode proj
"""
import os, argparse, json
from pathlib import Path
import numpy as np

# Optional deps for OpenML fetching / preprocessing
try:
    from sklearn.datasets import fetch_openml
    from sklearn.compose import make_column_selector, ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
except Exception:
    fetch_openml = None

def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def load_bonus_p(bonus_x_path: str) -> int:
    X = np.load(bonus_x_path)
    return int(X.shape[1])

def to_float_matrix_from_array(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.dtype.kind not in "fi":
        X = X.astype(np.float32, copy=False)
    return X

def to_float_matrix_from_csv(csv_path: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Keep numeric columns only
    df = df.select_dtypes(include=["number"]).dropna()
    X = df.to_numpy(dtype=np.float32)
    return X

def fetch_openml_numeric(name: str) -> np.ndarray:
    """
    Fetch an OpenML dataset by name (optionally 'name:version'), return all-numeric matrix.
    - One-hot encodes categoricals.
    - Robust to ds.target being str / list / Series / None.
    """
    if fetch_openml is None:
        raise RuntimeError("scikit-learn with fetch_openml is required for --openml fetching.")

    # allow "credit-g:1" style to pin version
    if ":" in name:
        base, ver = name.split(":", 1)
        ver = int(ver)
    else:
        base, ver = name, None  # let sklearn pick default; or set ver=1 if you prefer

    ds = fetch_openml(name=base, version=ver, as_frame=True)
    if hasattr(ds, "frame") and ds.frame is not None:
        df = ds.frame.copy()

        # drop targets if present (normalize to list of column names)
        drop_cols = []
        tgt = getattr(ds, "target", None)
        if tgt is not None:
            if isinstance(tgt, (list, tuple)):
                drop_cols.extend([t for t in tgt if isinstance(t, str) and t in df.columns])
            else:
                # tgt can be a pandas Series in some builds
                try:
                    tgt_str = str(tgt)
                    if tgt_str in df.columns:
                        drop_cols.append(tgt_str)
                except Exception:
                    pass
        if drop_cols:
            df = df.drop(columns=list(set(drop_cols)), errors="ignore")

        # numeric + categorical pipeline -> dense float matrix
        from sklearn.compose import make_column_selector, ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        num_sel = make_column_selector(dtype_include=["number"])
        cat_sel = make_column_selector(dtype_exclude=["number"])
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler(with_mean=True, with_std=True))])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        pre = ColumnTransformer([("num", num_pipe, num_sel),
                                 ("cat", cat_pipe, cat_sel)])
        X = pre.fit_transform(df).astype(np.float32, copy=False)
        return X

    # fallback to array API
    X = to_float_matrix_from_array(ds.data)
    return X


def match_dimension(X: np.ndarray, p_target: int, mode: str = "none", seed: int = 2025) -> np.ndarray:
    """
    Make X have exactly p_target columns.
    - "none" : do nothing; raise if mismatch.
    - "pca"  : PCA to p_target if p > p_target; if p < p_target, pad with zeros.
    - "proj" : Random Gaussian projection to p_target (Johnson–Lindenstrauss style).
    """
    n, p = X.shape
    if p == p_target or mode == "none":
        if p != p_target:
            raise ValueError(f"Feature dim mismatch: got p={p}, want p_target={p_target}. Use --match_mode pca|proj.")
        return X

    if mode == "pca":
        if p >= p_target:
            # reduce by PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=p_target, random_state=seed)
            Xr = pca.fit_transform(X)
            return Xr.astype(np.float32)
        else:
            # pad with zeros to the right
            pad = np.zeros((n, p_target - p), dtype=np.float32)
            return np.hstack([X.astype(np.float32), pad])

    if mode == "proj":
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1.0 / np.sqrt(p), size=(p, p_target)).astype(np.float32)
        Xr = X.astype(np.float32) @ W
        return Xr

    raise ValueError(f"Unknown match_mode: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonus_x", type=str, required=True,
                    help="Path to Bonus X.npy (to infer target p).")
    # Source choices
    ap.add_argument("--openml", type=str, nargs="*", default=[],
                    help="OpenML dataset names to fetch (e.g., credit-g adult).")
    ap.add_argument("--local_csv", type=str, default="",
                    help="Path to a local CSV to convert to numeric matrix.")
    # Output targets
    ap.add_argument("--save_npy", type=str, required=True,
                    help="Path to save a numeric .npy")
    ap.add_argument("--save_csv", type=str, required=True,
                    help="Path to save a numeric .csv")
    # Dimensional alignment
    ap.add_argument("--match_mode", type=str, default="pca", choices=["none", "pca", "proj"],
                    help="How to match feature dim to Bonus p.")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    p_target = load_bonus_p(args.bonus_x)

    # 1) Build a candidate pool (X matrices), from OpenML and/or local CSV
    X_pool = []

    if args.openml:
        for name in args.openml:
            print(f"[info] fetching from OpenML: {name}")
            X = fetch_openml_numeric(name)
            X = match_dimension(X, p_target=p_target, mode=args.match_mode, seed=args.seed)  # ★추가
            X_pool.append(X)

    if args.local_csv:
        print(f"[info] loading local CSV: {args.local_csv}")
        X = to_float_matrix_from_csv(args.local_csv)
        X = match_dimension(X, p_target=p_target, mode=args.match_mode, seed=args.seed)  # ★추가
        X_pool.append(X)

    if not X_pool:
        raise RuntimeError("No sources provided. Use --openml and/or --local_csv.")

    # 2) Concatenate rows from all sources (row-wise); then dim-match to p_target
    X_matched = np.concatenate(X_pool, axis=0).astype(np.float32)
    print(f"[info] pooled base shape after per-source match: {X_matched.shape} (target p={p_target})")
    # X_cat = np.concatenate(X_pool, axis=0).astype(np.float32)
    # print(f"[info] pooled base shape before match: {X_cat.shape}")
    #
    # X_matched = match_dimension(X_cat, p_target=p_target, mode=args.match_mode, seed=args.seed)
    # print(f"[info] base shape after match ({args.match_mode}): {X_matched.shape} (target p={p_target})")

    # 3) Save to requested outputs
    ensure_dir(args.save_npy); ensure_dir(args.save_csv)
    np.save(args.save_npy, X_matched.astype(np.float32))
    # Save CSV
    import pandas as pd
    pd.DataFrame(X_matched).to_csv(args.save_csv, index=False)
    print(f"[OK] saved .npy -> {args.save_npy}")
    print(f"[OK] saved .csv -> {args.save_csv}")

if __name__ == "__main__":
    main()
