# CausalPFN-DML/benchmarks/bonus/data/02_make_prior.py

## when running below, edit configurations :  --mu_mode columns --xi 0.5 --overlap medium
## this way, it's the closest to the paper's prior construction method
## if wanting to include openML sythetic data like what the paper did
## --mu_mode columns --xi 0.5 --overlap medium --extra_base /home/yesong/CausalPFN-DML/external/openml/credit_X.npy,/home/yesong/CausalPFN-DML/external/uci/census_X.csv

import os, json, argparse
import numpy as np
from pathlib import Path
import csv

# ---------------- Utils ----------------
def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[4] if len(start.parents) >= 5 else start.parents[-1]

def standardize(X, mean, std, eps=1e-8):
    return (X - mean) / (std + eps)

def load_X_from_path(p: Path) -> np.ndarray:
    """Load a table as numpy array of floats.
    Supports: .npy (dense matrix), .csv (all-numeric columns assumed or coercible).
    """
    if p.suffix == ".npy":
        X = np.load(p)
        return X.astype(np.float32)
    elif p.suffix == ".csv":
        # Lightweight CSV reader -> float matrix (non-numeric cells will raise).
        rows = []
        with open(p, newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                try:
                    rows.append([float(x) for x in r])
                except Exception:
                    # skip header or non-numeric rows
                    continue
        X = np.asarray(rows, dtype=np.float32)
        return X
    else:
        raise ValueError(f"Unsupported file extension for base table: {p}")

# ---------------- Feature Maps ----------------
def rff_features(X, n_feat=128, seed=None):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.normal(scale=1.0, size=(d, n_feat))
    b = rng.uniform(0, 2*np.pi, size=(n_feat,))
    Z = np.sqrt(2.0/n_feat) * np.cos(X @ W + b)
    return Z

def mlp_features(X, widths=(64, 64), seed=None):
    rng = np.random.default_rng(seed)
    H = X
    for w in widths:
        W = rng.normal(scale=1.0/np.sqrt(H.shape[1]), size=(H.shape[1], w))
        b = rng.normal(scale=0.1, size=(w,))
        H = np.tanh(H @ W + b)
    return H

# ---------------- DGP families (μ via functions) ----------------
def make_mu_family_functions(Xs, fam, rng):
    n, d = Xs.shape
    if fam == "linear":
        w0 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        w1 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        mu0 = Xs @ w0
        mu1 = mu0 + 1.0 + (Xs @ w1)
    elif fam == "poly":
        A = rng.normal(0, 0.2, size=(d, d)); A = (A + A.T) / 2.0
        quad = np.einsum("ni,ij,nj->n", Xs, A, Xs)
        w0 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        w1 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        mu0 = 0.7 * quad + Xs @ w0
        mu1 = mu0 + 1.0 + 0.7 * quad + Xs @ w1
    elif fam == "rff":
        Z1 = rff_features(Xs, n_feat=128, seed=rng.integers(1<<30))
        Z2 = rff_features(Xs, n_feat=128, seed=rng.integers(1<<30))
        w0 = rng.normal(0, 1, size=(Z1.shape[1],))
        w1 = rng.normal(0, 1, size=(Z2.shape[1],))
        mu0 = 2.0 * np.tanh(Z1 @ w0)
        mu1 = mu0 + 1.0 + 1.5 * np.tanh(Z2 @ w1)
    else:  # 'mlp'
        H1 = mlp_features(Xs, widths=(64, 64), seed=rng.integers(1<<30))
        H2 = mlp_features(Xs, widths=(64, 64), seed=rng.integers(1<<30))
        w0 = rng.normal(0, 1, size=(H1.shape[1],))
        w1 = rng.normal(0, 1, size=(H2.shape[1],))
        mu0 = 1.5 * np.tanh(H1 @ w0)
        mu1 = mu0 + 1.0 + 1.5 * np.tanh(H2 @ w1)
    return mu0, mu1

# ---------------- μ via column relabeling (paper-like) ----------------
def make_mu_family_columns(X_raw, rng):
    """Pick two distinct numeric columns from the *raw-scale* base table, treat them as μ0, μ1.
    Optionally apply small centering/scale to keep ranges reasonable.
    """
    n, d = X_raw.shape
    if d < 2:
        raise ValueError("Base table needs at least 2 numeric columns for 'columns' μ-mode.")
    j, k = rng.choice(d, size=2, replace=False)
    mu0 = X_raw[:, j].astype(np.float32)
    mu1 = X_raw[:, k].astype(np.float32)
    # light normalization to avoid extreme scales (center and divide by robust scale)
    mu0 = (mu0 - np.median(mu0)) / (np.std(mu0) + 1e-8)
    mu1 = (mu1 - np.median(mu1)) / (np.std(mu1) + 1e-8)
    # add a small shift so μ1 ≠ μ0 on average (keeps HTE signal)
    mu1 = mu1 + 0.5
    return mu0, mu1, (j, k)

# ---------------- Propensity with positivity weakening (Eq. 45 style) ----------------
def make_propensity(Xs, overlap, rng, xi=1.0):
    """e(x) = xi * sigmoid(f) + (1-xi) * 1{f>0}; then clamp to [0.02, 0.98].
    - overlap controls the temperature on f (strong<medium<weak).
    - xi ∈ [0,1]: xi↓ weakens positivity (more mass near 0/1).
    """
    temp = {"strong": 0.7, "medium": 1.2, "weak": 2.0}.get(overlap, 1.2)
    d = Xs.shape[1]
    g = rng.normal(0, 1/np.sqrt(d), size=(d,))
    f = (Xs @ g) * temp + rng.normal(0, 0.3)  # random intercept
    sigm = 1.0 / (1.0 + np.exp(-f))
    hard = (f > 0).astype(np.float32)
    e = xi * sigm + (1.0 - xi) * hard
    e = np.clip(e, 0.02, 0.98)
    return e

# ---------------- One shard ----------------
def make_shard_from_base(X_base, n, rng, mean, std,
                         mu_mode="functions", families=None,
                         noise=1.0, overlap="medium", xi=1.0):
    """Create one shard given a chosen base table X_base."""
    idx = rng.integers(0, X_base.shape[0], size=n)
    X_raw = X_base[idx].astype(np.float32)
    Xs = standardize(X_raw, mean, std)

    if mu_mode == "columns":
        mu0, mu1, used_cols = make_mu_family_columns(X_raw, rng)
        fam_tag = f"columns:{used_cols[0]},{used_cols[1]}"
    else:
        fam = rng.choice(families)
        mu0, mu1 = make_mu_family_functions(Xs, fam, rng)
        fam_tag = f"functions:{fam}"

    # heteroskedastic noise
    sig = np.exp(0.2 * (Xs @ rng.normal(0, 1/np.sqrt(Xs.shape[1]), size=(Xs.shape[1],))))
    sig = 0.5 + noise * np.clip(sig, 0.5, 3.0)

    e = make_propensity(Xs, overlap=overlap, rng=rng, xi=xi)
    T = (rng.random(n) < e).astype(np.float32)

    Y0 = mu0 + sig * rng.normal(size=n)
    Y1 = mu1 + sig * rng.normal(size=n)
    Y  = T * Y1 + (1 - T) * Y0

    return {
        "X": X_raw, "T": T.astype(np.float32), "Y": Y.astype(np.float32),
        "mu0_true": mu0.astype(np.float32), "mu1_true": mu1.astype(np.float32),
        "tau_true": (mu1 - mu0).astype(np.float32), "e_true": e.astype(np.float32),
        "fam": fam_tag
    }

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Prior simulation closer to paper (Figure 4).")
    # Base table pool
    ap.add_argument("--bonus_dir", type=str,
                    default="notebooks/bonus_benchmarks/data/bonus",
                    help="Folder containing Bonus X.npy (and optionally T.npy,Y.npy).")
    ap.add_argument("--extra_base", type=str, default="",
                    help="Comma-separated paths to extra base tables (X.npy or .csv).")
    # Output
    ap.add_argument("--out_dir", type=str, default="data/prior/bonus",
                    help="Output folder for prior shards.")
    # Sizes
    ap.add_argument("--shards", type=int, default=50)
    ap.add_argument("--n_per_shard", type=int, default=4096)
    # DGP options
    ap.add_argument("--mu_mode", type=str, default="functions",
                    choices=["functions", "columns"],
                    help="μ-generation mode: 'functions' (default) or 'columns' (paper-like relabel).")
    ap.add_argument("--families", type=str, default="linear,poly,rff,mlp",
                    help="Comma-separated families (used only if mu_mode=functions).")
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--overlap", type=str, default="medium",
                    choices=["strong","medium","weak"])
    ap.add_argument("--xi", type=float, default=1.0,
                    help="Positivity weakening mix ∈[0,1]; smaller=weaker positivity (Eq.45 style).")
    # Misc
    ap.add_argument("--bins_pct", type=float, default=0.995,
                    help="Percentile for global μ-histogram range.")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    # Resolve paths
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    if not os.path.isabs(args.bonus_dir):
        args.bonus_dir = str((repo_root / args.bonus_dir).resolve())
    if not os.path.isabs(args.out_dir):
        args.out_dir = str((repo_root / args.out_dir).resolve())
    os.makedirs(args.out_dir, exist_ok=True)

    # Build base table pool
    base_paths = []
    bonus_X_path = Path(args.bonus_dir) / "X.npy"
    if bonus_X_path.exists():
        base_paths.append(bonus_X_path)
    if args.extra_base.strip():
        for p in args.extra_base.split(","):
            p = p.strip()
            if not p:
                continue
            q = Path(p)
            if not q.is_absolute():
                q = (repo_root / q).resolve()
            if not q.exists():
                raise FileNotFoundError(f"Extra base table not found: {q}")
            base_paths.append(q)

    if not base_paths:
        raise FileNotFoundError("No base tables found. Provide Bonus X.npy or --extra_base with .npy/.csv files.")

    # Precompute mean/std from a reference base (use Bonus if available; otherwise first extra)
    X_ref = load_X_from_path(base_paths[0])
    mean = X_ref.mean(axis=0); std = X_ref.std(axis=0, ddof=0)

    # Families for functions mode
    fam_list = [s.strip() for s in args.families.split(",") if s.strip()]

    # Save meta
    meta = dict(
        base_paths=[str(p) for p in base_paths],
        out_dir=args.out_dir, shards=args.shards, n_per_shard=args.n_per_shard,
        mu_mode=args.mu_mode, families=fam_list,
        noise=args.noise, overlap=args.overlap, xi=args.xi,
        seed=args.seed, bins_pct=args.bins_pct,
    )
    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[info] repo_root = {repo_root}")
    print(f"[info] base pool size = {len(base_paths)}")
    print(f"[info] Saved meta to {Path(args.out_dir) / 'meta.json'}")

    rng = np.random.default_rng(args.seed)
    mu_collect = []

    for k in range(args.shards):
        # Randomly choose a base table for this shard (diversify base sources)
        base_p = Path(base_paths[rng.integers(0, len(base_paths))])
        X_base = load_X_from_path(base_p)

        shard = make_shard_from_base(
            X_base=X_base, n=args.n_per_shard, rng=np.random.default_rng(int(rng.integers(1<<30))),
            mean=mean, std=std,
            mu_mode=args.mu_mode, families=fam_list,
            noise=args.noise, overlap=args.overlap, xi=args.xi
        )
        out_path = Path(args.out_dir) / f"shard_{k:04d}.npz"
        np.savez_compressed(out_path, **{kk: vv for kk, vv in shard.items() if kk != "fam"})

        # Collect μ for global bin range suggestion
        sel = rng.integers(0, len(shard["mu0_true"]), size=min(2048, len(shard["mu0_true"])))
        mu_collect.append(shard["mu0_true"][sel]); mu_collect.append(shard["mu1_true"][sel])

        if (k + 1) % 10 == 0 or k == args.shards - 1:
            print(f"[info] Wrote {k+1}/{args.shards} … last: {out_path} (base={base_p.name})")

    mu_all = np.concatenate(mu_collect)
    lo = np.quantile(mu_all, (1 - args.bins_pct) / 2)
    hi = np.quantile(mu_all, 1 - (1 - args.bins_pct) / 2)
    bins_meta = {"mu_hist_range": [float(lo), float(hi)], "pct": args.bins_pct}
    with open(Path(args.out_dir) / "bins_meta.json", "w") as f:
        json.dump(bins_meta, f, ensure_ascii=False, indent=2)
    print(f"[info] Saved bins_meta to {Path(args.out_dir) / 'bins_meta.json'}")
    print(f"[OK] Finished. Wrote {args.shards} shards to {args.out_dir}")

if __name__ == "__main__":
    main()


#########################################################
# import os, json, argparse
# import numpy as np
# from pathlib import Path
#
# # ---------------- Utils ----------------
# def find_repo_root(start: Path) -> Path:
#     # Walk upwards from `start` to find a directory containing a `.git` folder.
#     # If found, treat that directory as the repository root.
#     for p in [start] + list(start.parents):
#         if (p / ".git").exists():
#             return p
#     # Fallback if no `.git` is found: choose a higher-level parent conservatively.
#     return start.parents[4] if len(start.parents) >= 5 else start.parents[-1]
#
# def standardize(X, mean, std, eps=1e-8):
#     # Standardize features using provided mean and std; guard against division by zero with eps.
#     return (X - mean) / (std + eps)
#
# # ---------------- Feature Maps ----------------
# def rff_features(X, n_feat=128, seed=None):
#     # Random Fourier Features (cosine) to create nonlinear embeddings of X.
#     rng = np.random.default_rng(seed)
#     d = X.shape[1]
#     W = rng.normal(scale=1.0, size=(d, n_feat))      # Gaussian projection matrix
#     b = rng.uniform(0, 2*np.pi, size=(n_feat,))      # Random phase offsets
#     Z = np.sqrt(2.0/n_feat) * np.cos(X @ W + b)      # Cosine-projected features
#     return Z
#
# def mlp_features(X, widths=(64, 64), seed=None):
#     # Lightweight random MLP feature map (non-trainable here).
#     rng = np.random.default_rng(seed)
#     H = X
#     for w in widths:
#         # Xavier-like scaling for random weights; tanh nonlinearity.
#         W = rng.normal(scale=1.0/np.sqrt(H.shape[1]), size=(H.shape[1], w))
#         b = rng.normal(scale=0.1, size=(w,))
#         H = np.tanh(H @ W + b)
#     return H  # Shape: [n, widths[-1]]
#
# # ---------------- DGP families ----------------
# def make_mu_family(X, fam, rng):
#     """
#     Generate potential outcome functions μ0(x), μ1(x) with heterogeneous treatment effects.
#     fam in {'linear','poly','rff','mlp'} selects the functional family.
#     """
#     n, d = X.shape
#     if fam == "linear":
#         # Linear base: μ1 differs from μ0 by a constant shift + a different linear component.
#         w0 = rng.normal(0, 1/np.sqrt(d), size=(d,))
#         w1 = rng.normal(0, 1/np.sqrt(d), size=(d,))
#         mu0 = X @ w0
#         mu1 = mu0 + 1.0 + (X @ w1)  # induces HTE
#     elif fam == "poly":
#         # Quadratic form + low-rank interactions to add nonlinearity.
#         A = rng.normal(0, 0.2, size=(d, d))
#         A = (A + A.T) / 2.0                         # symmetrize
#         quad = np.einsum("ni,ij,nj->n", X, A, X)    # xᵀ A x for each sample
#         w0 = rng.normal(0, 1/np.sqrt(d), size=(d,))
#         w1 = rng.normal(0, 1/np.sqrt(d), size=(d,))
#         mu0 = 0.7 * quad + X @ w0
#         mu1 = mu0 + 1.0 + 0.7 * quad + X @ w1
#     elif fam == "rff":
#         # Nonlinear RFF maps for μ0 and μ1 to create structural differences.
#         Z1 = rff_features(X, n_feat=128, seed=rng.integers(1<<30))
#         Z2 = rff_features(X, n_feat=128, seed=rng.integers(1<<30))
#         w0 = rng.normal(0, 1, size=(Z1.shape[1],))
#         w1 = rng.normal(0, 1, size=(Z2.shape[1],))
#         mu0 = 2.0 * np.tanh(Z1 @ w0)
#         mu1 = mu0 + 1.0 + 1.5 * np.tanh(Z2 @ w1)
#     else:  # 'mlp'
#         # Random MLP embeddings followed by tanh to form μ0/μ1.
#         H1 = mlp_features(X, widths=(64, 64), seed=rng.integers(1<<30))
#         H2 = mlp_features(X, widths=(64, 64), seed=rng.integers(1<<30))
#         w0 = rng.normal(0, 1, size=(H1.shape[1],))
#         w1 = rng.normal(0, 1, size=(H2.shape[1],))
#         mu0 = 1.5 * np.tanh(H1 @ w0)
#         mu1 = mu0 + 1.0 + 1.5 * np.tanh(H2 @ w1)
#     return mu0, mu1
#
# def make_propensity(X, overlap, rng):
#     # Construct propensity e(x) with a temperature to control overlap.
#     # Add a random intercept so the overall treatment rate varies across shards.
#     temp = {"strong": 0.7, "medium": 1.2, "weak": 2.0}.get(overlap, 1.2)
#     d = X.shape[1]
#     g = rng.normal(0, 1/np.sqrt(d), size=(d,))
#     logit = (X @ g) * temp + rng.normal(0, 0.3)  # random intercept
#     e = 1.0 / (1.0 + np.exp(-logit))
#     e = np.clip(e, 0.02, 0.98)                   # enforce positivity (avoid extremes)
#     return e
#
# # ---------------- Shard maker ----------------
# def make_shard(X_ref, mean, std, families, n=4096, noise=1.0, overlap="medium", seed=0):
#     # Create one shard:
#     # 1) resample X from Bonus-distribution reference;
#     # 2) standardize with global mean/std to align scales;
#     # 3) sample μ0/μ1 from a random family; 4) heteroskedastic noise;
#     # 5) sample T via propensity e(x); 6) assemble observed Y.
#     rng = np.random.default_rng(seed)
#     idx = rng.integers(0, X_ref.shape[0], size=n)
#     X = X_ref[idx].astype(np.float32)
#     Xs = standardize(X, mean, std)  # standardized view used by DGPs
#
#     # Choose a functional family at random and generate μ0/μ1 (ensures HTE).
#     fam = rng.choice(families)
#     mu0, mu1 = make_mu_family(Xs, fam, rng)
#
#     # Heteroskedastic noise: σ(x) ≈ log-normal-like, then clipped and scaled.
#     sig = np.exp(0.2 * (Xs @ rng.normal(0, 1/np.sqrt(Xs.shape[1]), size=(Xs.shape[1],))))
#     sig = 0.5 + noise * np.clip(sig, 0.5, 3.0)
#
#     # Treatment assignment via e(x); then realize observed outcome.
#     e = make_propensity(Xs, overlap, rng)
#     T = (rng.random(n) < e).astype(np.float32)
#     Y0 = mu0 + sig * rng.normal(size=n)
#     Y1 = mu1 + sig * rng.normal(size=n)
#     Y  = T * Y1 + (1 - T) * Y0
#
#     return {
#         "X": X.astype(np.float32),         # keep raw-scale X for flexibility downstream
#         "T": T.astype(np.float32),
#         "Y": Y.astype(np.float32),
#         "mu0_true": mu0.astype(np.float32),
#         "mu1_true": mu1.astype(np.float32),
#         "tau_true": (mu1 - mu0).astype(np.float32),
#         "e_true": e.astype(np.float32),
#         "fam": fam,                        # family label (for logging/diagnostics)
#     }
#
# # ---------------- Main ----------------
# def main():
#     # CLI: configure prior-simulation settings closer to the paper’s spirit.
#     ap = argparse.ArgumentParser(description="Prior simulation closer to paper settings.")
#     ap.add_argument("--bonus_dir", type=str,
#                     default="notebooks/bonus_benchmarks/data/bonus",
#                     help="Folder containing X.npy, T.npy, Y.npy.")
#     ap.add_argument("--out_dir", type=str, default="data/prior/bonus",
#                     help="Output folder for prior shards (.npz).")
#     ap.add_argument("--shards", type=int, default=50)          # number of shards
#     ap.add_argument("--n_per_shard", type=int, default=4096)   # rows per shard
#     ap.add_argument("--noise", type=float, default=1.0)        # outcome noise scale
#     ap.add_argument("--overlap", type=str, default="medium",
#                     choices=["strong","medium","weak"])        # overlap strength
#     ap.add_argument("--seed", type=int, default=2025)          # global seed
#     ap.add_argument("--families", type=str,
#                     default="linear,poly,rff,mlp",
#                     help="Comma-separated list of DGP families to sample from.")
#     ap.add_argument("--bins_pct", type=float, default=0.995,
#                     help="Percentile for global μ-histogram range suggestion.")
#     args = ap.parse_args()
#
#     # Resolve paths relative to the repo root (discovered via `.git`).
#     here = Path(__file__).resolve()
#     repo_root = find_repo_root(here)
#     if not os.path.isabs(args.bonus_dir): args.bonus_dir = str((repo_root / args.bonus_dir).resolve())
#     if not os.path.isabs(args.out_dir):   args.out_dir   = str((repo_root / args.out_dir).resolve())
#     os.makedirs(args.out_dir, exist_ok=True)
#
#     # Load Bonus X reference to mimic covariate distribution in prior simulation.
#     X_path = Path(args.bonus_dir) / "X.npy"
#     print(f"[info] repo_root = {repo_root}")
#     print(f"[info] Looking for X.npy at: {X_path}")
#     if not X_path.exists():
#         # Fail-fast with a clear message if X.npy is missing.
#         raise FileNotFoundError(f"Not found: {X_path}")
#
#     X_ref = np.load(X_path)
#     # Estimate mean/std from Bonus features to standardize simulated DGPs accordingly.
#     mean = X_ref.mean(axis=0)
#     std  = X_ref.std(axis=0, ddof=0)
#     # Parse the selected DGP families.
#     fam_list = [s.strip() for s in args.families.split(",") if s.strip()]
#
#     # Save run metadata for reproducibility and downstream inspection.
#     meta = dict(
#         bonus_dir=args.bonus_dir, out_dir=args.out_dir,
#         shards=args.shards, n_per_shard=args.n_per_shard,
#         noise=args.noise, overlap=args.overlap, seed=args.seed,
#         families=fam_list
#     )
#     with open(Path(args.out_dir) / "meta.json", "w") as f:
#         json.dump(meta, f, ensure_ascii=False, indent=2)
#     print(f"[info] Saved meta to {Path(args.out_dir) / 'meta.json'}")
#
#     rng = np.random.default_rng(args.seed)
#     mu_collect = []  # collect μ samples to determine a stable global histogram range
#     for k in range(args.shards):
#         # Generate one shard (resample→standardize→DGP→package arrays).
#         shard = make_shard(
#             X_ref=X_ref, mean=mean, std=std, families=fam_list,
#             n=args.n_per_shard, noise=args.noise,
#             overlap=args.overlap, seed=int(rng.integers(1<<30))
#         )
#         # Save shard as compressed .npz (friendly for concatenation/streaming later).
#         out_path = Path(args.out_dir) / f"shard_{k:04d}.npz"
#         np.savez_compressed(out_path, **{k: v for k, v in shard.items() if k != "fam"})
#         # Subsample μ0/μ1 for global range estimation (memory-friendly).
#         sel = rng.integers(0, len(shard["mu0_true"]), size=min(2048, len(shard["mu0_true"])))
#         mu_collect.append(shard["mu0_true"][sel]); mu_collect.append(shard["mu1_true"][sel])
#         if (k + 1) % 10 == 0 or k == args.shards - 1:
#             print(f"[info] Wrote {k+1}/{args.shards} … last: {out_path}")
#
#     # Suggest a global histogram range for μ-heads in the PFN (robust to outliers).
#     mu_all = np.concatenate(mu_collect)
#     lo = np.quantile(mu_all, (1 - args.bins_pct) / 2)                # lower percentile
#     hi = np.quantile(mu_all, 1 - (1 - args.bins_pct) / 2)            # upper percentile
#     bins_meta = {"mu_hist_range": [float(lo), float(hi)], "pct": args.bins_pct}
#     with open(Path(args.out_dir) / "bins_meta.json", "w") as f:
#         json.dump(bins_meta, f, ensure_ascii=False, indent=2)
#     print(f"[info] Saved bins_meta to {Path(args.out_dir) / 'bins_meta.json'}")
#     print(f"[OK] Finished. Wrote {args.shards} shards to {args.out_dir}")
#
# if __name__ == "__main__":
#     main()


