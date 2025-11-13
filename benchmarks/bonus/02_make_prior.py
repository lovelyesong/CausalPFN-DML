# CausalPFN-DML/benchmarks/bonus/data/02_make_prior.py
import os, json, argparse
import numpy as np
from pathlib import Path

# ---------------- Utils ----------------
def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[4] if len(start.parents) >= 5 else start.parents[-1]

def standardize(X, mean, std, eps=1e-8):
    return (X - mean) / (std + eps)

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
    return H  # [n, widths[-1]]

# ---------------- DGP families ----------------
def make_mu_family(X, fam, rng):
    """
    Return mu0(x), mu1(x) with heterogeneous effect.
    fam in {'linear','poly','rff','mlp'}
    """
    n, d = X.shape
    if fam == "linear":
        w0 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        w1 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        mu0 = X @ w0
        mu1 = mu0 + 1.0 + (X @ w1)  # HTE
    elif fam == "poly":
        # quadratic + interactions (low-rank)
        A = rng.normal(0, 0.2, size=(d, d))
        A = (A + A.T) / 2.0
        quad = np.einsum("ni,ij,nj->n", X, A, X)
        w0 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        w1 = rng.normal(0, 1/np.sqrt(d), size=(d,))
        mu0 = 0.7 * quad + X @ w0
        mu1 = mu0 + 1.0 + 0.7 * quad + X @ w1
    elif fam == "rff":
        Z1 = rff_features(X, n_feat=128, seed=rng.integers(1<<30))
        Z2 = rff_features(X, n_feat=128, seed=rng.integers(1<<30))
        w0 = rng.normal(0, 1, size=(Z1.shape[1],))
        w1 = rng.normal(0, 1, size=(Z2.shape[1],))
        mu0 = 2.0 * np.tanh(Z1 @ w0)
        mu1 = mu0 + 1.0 + 1.5 * np.tanh(Z2 @ w1)
    else:  # 'mlp'
        H1 = mlp_features(X, widths=(64, 64), seed=rng.integers(1<<30))
        H2 = mlp_features(X, widths=(64, 64), seed=rng.integers(1<<30))
        w0 = rng.normal(0, 1, size=(H1.shape[1],))
        w1 = rng.normal(0, 1, size=(H2.shape[1],))
        mu0 = 1.5 * np.tanh(H1 @ w0)
        mu1 = mu0 + 1.0 + 1.5 * np.tanh(H2 @ w1)
    return mu0, mu1

def make_propensity(X, overlap, rng):
    temp = {"strong": 0.7, "medium": 1.2, "weak": 2.0}.get(overlap, 1.2)
    d = X.shape[1]
    g = rng.normal(0, 1/np.sqrt(d), size=(d,))
    logit = (X @ g) * temp + rng.normal(0, 0.3)  # random intercept
    e = 1.0 / (1.0 + np.exp(-logit))
    e = np.clip(e, 0.02, 0.98)
    return e

# ---------------- Shard maker ----------------
def make_shard(X_ref, mean, std, families, n=4096, noise=1.0, overlap="medium", seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X_ref.shape[0], size=n)
    X = X_ref[idx].astype(np.float32)
    Xs = standardize(X, mean, std)  # align to Bonus scale

    fam = rng.choice(families)
    mu0, mu1 = make_mu_family(Xs, fam, rng)

    # heteroskedastic noise
    sig = np.exp(0.2 * (Xs @ rng.normal(0, 1/np.sqrt(Xs.shape[1]), size=(Xs.shape[1],))))  # ~ log-normal
    sig = 0.5 + noise * np.clip(sig, 0.5, 3.0)

    e = make_propensity(Xs, overlap, rng)
    T = (rng.random(n) < e).astype(np.float32)
    Y0 = mu0 + sig * rng.normal(size=n)
    Y1 = mu1 + sig * rng.normal(size=n)
    Y  = T * Y1 + (1 - T) * Y0

    return {
        "X": X.astype(np.float32),         # raw X (keep original scale)
        "T": T.astype(np.float32),
        "Y": Y.astype(np.float32),
        "mu0_true": mu0.astype(np.float32),
        "mu1_true": mu1.astype(np.float32),
        "tau_true": (mu1 - mu0).astype(np.float32),
        "e_true": e.astype(np.float32),
        "fam": fam,
    }

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Prior simulation closer to paper settings.")
    ap.add_argument("--bonus_dir", type=str,
                    default="notebooks/bonus_benchmarks/data/bonus",
                    help="Folder containing X.npy, T.npy, Y.npy.")
    ap.add_argument("--out_dir", type=str, default="data/prior/bonus",
                    help="Output folder for prior shards.")
    ap.add_argument("--shards", type=int, default=50)
    ap.add_argument("--n_per_shard", type=int, default=4096)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--overlap", type=str, default="medium",
                    choices=["strong","medium","weak"])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--families", type=str,
                    default="linear,poly,rff,mlp",
                    help="Comma-separated families to sample from.")
    ap.add_argument("--bins_pct", type=float, default=0.995,
                    help="Global range percentile for mu hist bins (e.g., 0.995).")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    if not os.path.isabs(args.bonus_dir): args.bonus_dir = str((repo_root / args.bonus_dir).resolve())
    if not os.path.isabs(args.out_dir):   args.out_dir   = str((repo_root / args.out_dir).resolve())
    os.makedirs(args.out_dir, exist_ok=True)

    X_path = Path(args.bonus_dir) / "X.npy"
    print(f"[info] repo_root = {repo_root}")
    print(f"[info] Looking for X.npy at: {X_path}")
    if not X_path.exists():
        raise FileNotFoundError(f"Not found: {X_path}")

    X_ref = np.load(X_path)
    mean = X_ref.mean(axis=0)
    std  = X_ref.std(axis=0, ddof=0)
    fam_list = [s.strip() for s in args.families.split(",") if s.strip()]

    # Save global meta
    meta = dict(
        bonus_dir=args.bonus_dir, out_dir=args.out_dir,
        shards=args.shards, n_per_shard=args.n_per_shard,
        noise=args.noise, overlap=args.overlap, seed=args.seed,
        families=fam_list
    )
    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[info] Saved meta to {Path(args.out_dir) / 'meta.json'}")

    rng = np.random.default_rng(args.seed)
    mu_collect = []  # for global bin range
    for k in range(args.shards):
        shard = make_shard(
            X_ref=X_ref, mean=mean, std=std, families=fam_list,
            n=args.n_per_shard, noise=args.noise,
            overlap=args.overlap, seed=int(rng.integers(1<<30))
        )
        out_path = Path(args.out_dir) / f"shard_{k:04d}.npz"
        np.savez_compressed(out_path, **{k: v for k, v in shard.items() if k != "fam"})
        # collect small subsample for global bin range
        sel = rng.integers(0, len(shard["mu0_true"]), size=min(2048, len(shard["mu0_true"])))
        mu_collect.append(shard["mu0_true"][sel]); mu_collect.append(shard["mu1_true"][sel])
        if (k + 1) % 10 == 0 or k == args.shards - 1:
            print(f"[info] Wrote {k+1}/{args.shards} … last: {out_path}")

    # Global histogram range suggestion (for PPD heads)
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

############################################
# import os, json, argparse
# import numpy as np
# from pathlib import Path
#
# def rff_features(X, n_feat=128, seed=None):
#     rng = np.random.default_rng(seed)
#     d = X.shape[1]
#     W = rng.normal(scale=1.0, size=(d, n_feat))
#     b = rng.uniform(0, 2*np.pi, size=(n_feat,))
#     Z = np.sqrt(2.0/n_feat) * np.cos(X @ W + b)
#     return Z
#
# def sample_from_Xref(X_ref, n, rng):
#     idx = rng.integers(0, X_ref.shape[0], size=n)
#     return X_ref[idx].astype(np.float32)
#
# def make_shard(X_ref, n=4096, noise=1.0, overlap="medium", seed=0):
#     """Return dict with X, T, Y, mu0_true, mu1_true, tau_true, e_true."""
#     rng = np.random.default_rng(seed)
#     X = sample_from_Xref(X_ref, n, rng)
#
#     # nonlinear bases
#     Z  = rff_features(X, n_feat=128, seed=rng.integers(1<<30))
#     Z2 = rff_features(X, n_feat=128, seed=rng.integers(1<<30))
#
#     # mu0, mu1
#     w0 = rng.normal(0, 1, size=Z.shape[1])
#     w1 = rng.normal(0, 1, size=Z2.shape[1])
#     mu0 = 2.0 * np.tanh(Z @ w0)
#     mu1 = mu0 + 1.0 + 1.5 * np.tanh(Z2 @ w1)  # heterogeneous effect
#
#     # propensity e(x) with overlap control
#     scale = {"strong": 0.75, "medium": 1.25, "weak": 2.0}.get(overlap, 1.25)
#     wg = rng.normal(0, 1, size=Z.shape[1])
#     logit = (Z @ wg) / np.sqrt(Z.shape[1]) * scale
#     e = 1.0 / (1.0 + np.exp(-logit))
#     e = np.clip(e, 0.02, 0.98)  # positivity
#
#     # observed outcome
#     T = (rng.random(n) < e).astype(np.float32)
#     Y0 = mu0 + noise * rng.normal(size=n)
#     Y1 = mu1 + noise * rng.normal(size=n)
#     Y  = T * Y1 + (1 - T) * Y0
#
#     return {
#         "X": X.astype(np.float32),
#         "T": T.astype(np.float32),
#         "Y": Y.astype(np.float32),
#         "mu0_true": mu0.astype(np.float32),
#         "mu1_true": mu1.astype(np.float32),
#         "tau_true": (mu1 - mu0).astype(np.float32),
#         "e_true": e.astype(np.float32),
#     }
#
# def find_repo_root(start: Path) -> Path:
#     """Walk up parents until a directory containing .git is found."""
#     for p in [start] + list(start.parents):
#         if (p / ".git").exists():
#             return p
#     # Fallback: go up to 4 levels (…/CausalPFN-DML expected)
#     return start.parents[4] if len(start.parents) >= 5 else start.parents[-1]
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Create prior-simulation shards using Bonus X distribution."
#     )
#     # ✅ your real X/T/Y default location
#     parser.add_argument(
#         "--bonus_dir",
#         type=str,
#         default="notebooks/bonus_benchmarks/data/bonus",
#         help="Folder containing X.npy, T.npy, Y.npy.",
#     )
#     parser.add_argument(
#         "--out_dir",
#         type=str,
#         default="data/prior/bonus",
#         help="Output folder for prior shards.",
#     )
#     parser.add_argument("--shards", type=int, default=50)
#     parser.add_argument("--n_per_shard", type=int, default=4096)
#     parser.add_argument("--noise", type=float, default=1.0)
#     parser.add_argument("--overlap", type=str, default="medium",
#                         choices=["strong","medium","weak"])
#     parser.add_argument("--seed", type=int, default=2025)
#     args = parser.parse_args()
#
#     here = Path(__file__).resolve()
#     repo_root = find_repo_root(here)
#     print(f"[info] repo_root = {repo_root}")
#
#     # Resolve to repo_root if relative
#     bonus_dir = Path(args.bonus_dir)
#     if not bonus_dir.is_absolute():
#         bonus_dir = repo_root / bonus_dir
#     out_dir = Path(args.out_dir)
#     if not out_dir.is_absolute():
#         out_dir = repo_root / out_dir
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     print(f"[info] resolved bonus_dir = {bonus_dir}")
#     print(f"[info] resolved out_dir   = {out_dir}")
#
#     X_ref_path = bonus_dir / "X.npy"
#     print(f"[info] Looking for X.npy at: {X_ref_path.resolve()}")
#     if not X_ref_path.exists():
#         raise FileNotFoundError(
#             f"Not found: {X_ref_path.resolve()}\n"
#             f"Fix: pass the correct folder via --bonus_dir, e.g.\n"
#             f"  --bonus_dir /home/yesong/CausalPFN-DML/notebooks/bonus_benchmarks/data/bonus"
#         )
#
#     X_ref = np.load(X_ref_path)
#     print(f"[info] Loaded X_ref with shape {X_ref.shape}")
#
#     rng = np.random.default_rng(args.seed)
#
#     # save global meta
#     meta = {
#         "bonus_dir": str(bonus_dir.resolve()),
#         "out_dir": str(out_dir.resolve()),
#         "shards": args.shards,
#         "n_per_shard": args.n_per_shard,
#         "noise": args.noise,
#         "overlap": args.overlap,
#         "seed": args.seed,
#     }
#     with open(out_dir / "meta.json", "w") as f:
#         json.dump(meta, f, ensure_ascii=False, indent=2)
#     print(f"[info] Saved meta to {out_dir / 'meta.json'}")
#
#     # generate shards
#     for k in range(args.shards):
#         shard = make_shard(
#             X_ref=X_ref,
#             n=args.n_per_shard,
#             noise=args.noise,
#             overlap=args.overlap,
#             seed=int(rng.integers(1<<30)),
#         )
#         out_path = out_dir / f"shard_{k:04d}.npz"
#         np.savez_compressed(out_path, **shard)
#         if (k + 1) % 10 == 0 or k == args.shards - 1:
#             print(f"[info] Wrote {k+1}/{args.shards} shards … last: {out_path}")
#
#     print(f"[OK] Finished. Wrote {args.shards} shards to {out_dir.resolve()}")
#
# if __name__ == "__main__":
#     main()
