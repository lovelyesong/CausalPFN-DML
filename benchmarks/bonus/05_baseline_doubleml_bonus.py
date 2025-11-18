# benchmarks/bonus/05_baseline_doubleml_bonus.py
# ---------------------------------------------------------
# Baseline using DoubleMLIRM with MLP learners on the BONUS data.
# For each seed:
#   - Fit DoubleMLIRM (MLP for outcome and propensity)
#   - Extract out-of-fold nuisance predictions (mu0_hat, mu1_hat, e_hat)
#   - Save them under: out_root/seed_<SEED>/
#   - Compute "paper" WAAE (DR/AIPW-based) for the DML baseline
#
# This script also aggregates WAAE over multiple seeds and writes:
#   out_root/05_dml_multi_summary.json
#
# The per-seed folders are compatible with 07_eval_waae_multi.py
# via: --dml_root <out_root>.
# ---------------------------------------------------------

# configurations

# --bonus_dir
# notebooks / bonus_benchmarks / data / bonus \
# - -out_root
# benchmarks / bonus / outputs_dml05 \
# - -seed_start
# 2025 \
# - -n_seeds
# 20 \
# - -n_folds
# 5 \
# - -n_rep
# 1

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import doubleml as dml
from sklearn.neural_network import MLPRegressor, MLPClassifier


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Find the git repository root (fallback: a few levels up)."""
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]


def load_bonus_arrays(bonus_dir: Path):
    """Load X, T, Y from BONUS directory."""
    X = np.load(bonus_dir / "X.npy").astype(np.float32)
    T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
    Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
    return X, T, Y


def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_anchor, mu1_anchor):
    r"""
    WAAE in the style of the r69 paper:

      WAAE(μ̂) = ∑_t | μ̂̄(t) - μ̄(t) | * P_N(t),

    where μ̄(t) is computed via DR/AIPW:

      μ̄(0) = (1/N) * Σ_i [ μ̂0(X_i) + 1{T_i=0}/(1 - ê(X_i)) * (Y_i - μ̂0(X_i)) ]
      μ̄(1) = (1/N) * Σ_i [ μ̂1(X_i) + 1{T_i=1}/ê(X_i)        * (Y_i - μ̂1(X_i)) ].

    Parameters
    ----------
    Y, T : np.ndarray
        Outcome and treatment indicators.
    mu0_pred, mu1_pred : np.ndarray
        Candidate predictions whose WAAE we want to evaluate.
    e_hat : np.ndarray
        Propensity score estimates ê(X).
    mu0_anchor, mu1_anchor : np.ndarray
        Anchor functions used for the AIPW target means (e.g., DML OOF nuisances).

    Returns
    -------
    dict
        Contains WAAE and intermediate summary statistics.
    """
    r1 = float((T == 1).mean())
    r0 = 1.0 - r1

    mu0_bar_hat = float(mu0_pred.mean())
    mu1_bar_hat = float(mu1_pred.mean())

    e1 = np.clip(e_hat, 1e-3, 1 - 1e-3)
    e0 = 1.0 - e1
    ind0 = (T == 0).astype(np.float32)
    ind1 = 1.0 - ind0

    aipw0 = mu0_anchor + ind0 / e0 * (Y - mu0_anchor)
    aipw1 = mu1_anchor + ind1 / e1 * (Y - mu1_anchor)
    mu0_bar = float(aipw0.mean())
    mu1_bar = float(aipw1.mean())

    err0 = abs(mu0_bar_hat - mu0_bar)
    err1 = abs(mu1_bar_hat - mu1_bar)
    waae = r0 * err0 + r1 * err1

    return {
        "waae": float(waae),
        "r0": float(r0),
        "r1": float(r1),
        "err0": float(err0),
        "err1": float(err1),
        "mu0_mean_pred": float(mu0_bar_hat),
        "mu1_mean_pred": float(mu1_bar_hat),
        "mu0_mean_true": float(mu0_bar),
        "mu1_mean_true": float(mu1_bar),
    }


def _extract_oof(pred_arr: np.ndarray) -> np.ndarray:
    """
    Extract an OOF-like prediction vector from DoubleMLIRM.predictions entry.

    In DoubleML, predictions["ml_g0"] / ["ml_g1"] / ["ml_m"] typically have
    shape (n_obs, n_folds * n_rep) or (n_obs, n_rep, n_folds), depending on version.
    For robustness, we handle:
      - 1D: already (n_obs,)
      - 2D: (n_obs, K) -> average over axis=1
      - 3D: (n_obs, K1, K2) -> average over axes=(1,2)

    If each observation has exactly one valid cross-fitted prediction per rep,
    the mean over folds/rep is equivalent to the OOF prediction.
    """
    arr = np.asarray(pred_arr)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        v = np.nanmean(arr, axis=1)
        return v.astype(np.float32)
    if arr.ndim == 3:
        v = np.nanmean(arr, axis=(1, 2))
        return v.astype(np.float32)
    raise ValueError(f"Unexpected prediction array shape: {arr.shape}")


def build_learners_mlp(seed: int):
    """
    Build MLP learners for DoubleMLIRM (outcome and propensity).

    This is an MLP-based baseline that roughly mimics a multi-layer architecture.
    You can adjust hidden_layer_sizes or other hyperparameters if needed.
    """
    ml_g = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed,
    )
    ml_m = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed + 1000,
    )
    return ml_g, ml_m


def run_one_seed(
    seed: int,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    out_root: Path,
    n_folds: int = 5,
    n_rep: int = 1,
):
    """
    Run DoubleMLIRM for a single seed, save nuisances, and compute WAAE.

    Steps:
      1. Set up DoubleMLData from X, Y, T.
      2. Build MLP learners and fit DoubleMLIRM with store_predictions=True.
      3. Extract OOF nuisances:
           - mu0_hat(x) (ml_g0)
           - mu1_hat(x) (ml_g1)
           - e_hat(x)   (ml_m)
      4. Save nuisances to out_root/seed_<seed>/.
      5. Compute paper-style WAAE for DML baseline (using its own nuisances as anchor).
      6. Save per-seed JSON summary.

    Returns
    -------
    float
        The WAAE value for this seed.
    """
    dml_data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=T)

    ml_g, ml_m = build_learners_mlp(seed)
    dml_irm = dml.DoubleMLIRM(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score="ATE",
    )

    t0 = time.time()
    print(f"[seed {seed}] Fitting DoubleMLIRM (MLP)…")
    dml_irm.fit(store_predictions=True)
    elapsed = time.time() - t0
    ate = float(dml_irm.coef[0])
    print(f"[seed {seed}] fit done in {elapsed:.1f}s, ATE={ate:.6f}")

    # Extract nuisance predictions
    pred_g0 = dml_irm.predictions["ml_g0"]
    pred_g1 = dml_irm.predictions["ml_g1"]
    pred_m  = dml_irm.predictions["ml_m"]

    mu0_hat = _extract_oof(pred_g0)
    mu1_hat = _extract_oof(pred_g1)
    e_hat   = _extract_oof(pred_m)

    assert len(mu0_hat) == len(X)
    assert len(mu1_hat) == len(X)
    assert len(e_hat)   == len(X)

    seed_dir = out_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    np.save(seed_dir / "mu0_hat.npy", mu0_hat.astype(np.float32))
    np.save(seed_dir / "mu1_hat.npy", mu1_hat.astype(np.float32))
    np.save(seed_dir / "e_hat.npy",   e_hat.astype(np.float32))

    # Paper-style WAAE for DML-only baseline (OOF nuisances = anchor)
    waae_stats = waae_paper(
        Y=Y,
        T=T,
        mu0_pred=mu0_hat,
        mu1_pred=mu1_hat,
        e_hat=e_hat,
        mu0_anchor=mu0_hat,
        mu1_anchor=mu1_hat,
    )

    ci = dml_irm.confint()
    ci_low = float(ci.iloc[0, 0])
    ci_high = float(ci.iloc[0, 1])

    seed_report = {
        "seed": seed,
        "ATE": ate,
        "SE": float(dml_irm.se[0]),
        "CI95": [ci_low, ci_high],
        "waae_paper": waae_stats,
        "timing_sec": float(elapsed),
        "n_folds": int(n_folds),
        "n_rep": int(n_rep),
    }

    with open(seed_dir / "05_dml_seed_summary.json", "w") as f:
        json.dump(seed_report, f, indent=2)

    print(f"[seed {seed}] WAAE(paper)={waae_stats['waae']:.6f}  -> saved npy & json")
    return waae_stats["waae"]


# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "BONUS baseline using DoubleMLIRM (MLP) across multiple seeds.\n"
            "For each seed, this script fits DoubleML, saves OOF mu0/mu1/e_hat, "
            "and computes paper-style WAAE for the DML baseline."
        )
    )
    parser.add_argument(
        "--bonus_dir",
        type=str,
        default="notebooks/bonus_benchmarks/data/bonus",
        help="Directory containing X.npy, T.npy, Y.npy.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="benchmarks/bonus/outputs_dml05",
        help="Output root under which seed_<seed>/ subfolders will be created.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=2025,
        help="First seed (e.g., 2025).",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=20,
        help="Number of seeds to run: seed_start, seed_start+1, ...",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds for DoubleMLIRM.",
    )
    parser.add_argument(
        "--n_rep",
        type=int,
        default=1,
        help="Number of DoubleML repetitions.",
    )

    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    bonus_dir = (repo_root / args.bonus_dir).resolve()
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] repo_root   = {repo_root}")
    print(f"[info] BONUS_DIR   = {bonus_dir}")
    print(f"[info] OUT_ROOT    = {out_root}")

    X, T, Y = load_bonus_arrays(bonus_dir)
    n, p = X.shape
    print(f"[info] X={X.shape}, T={T.shape}, Y={Y.shape}")

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    waae_list = []

    t0_all = time.time()
    for s in seeds:
        waae_s = run_one_seed(
            seed=s,
            X=X,
            T=T,
            Y=Y,
            out_root=out_root,
            n_folds=args.n_folds,
            n_rep=args.n_rep,
        )
        waae_list.append(waae_s)

    elapsed_all = time.time() - t0_all
    waae_arr = np.asarray(waae_list, dtype=float)

    agg = {
        "seeds": seeds,
        "waae_per_seed": [float(w) for w in waae_arr],
        "waae_mean": float(waae_arr.mean()),
        "waae_std": float(waae_arr.std(ddof=1)) if len(waae_arr) > 1 else 0.0,
        "elapsed_total_sec": float(elapsed_all),
        "n": int(n),
        "p": int(p),
        "n_folds": int(args.n_folds),
        "n_rep": int(args.n_rep),
    }

    summary_path = out_root / "05_dml_multi_summary.json"
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n[SUMMARY over seeds]")
    print(f"  seeds          : {seeds[0]} .. {seeds[-1]} (total {len(seeds)})")
    print(f"  WAAE mean      : {agg['waae_mean']:.6f}")
    print(f"  WAAE std       : {agg['waae_std']:.6f}")
    print(f"  total time (s) : {agg['elapsed_total_sec']:.1f}")
    print(f"[OK] saved -> {summary_path}")


if __name__ == "__main__":
    main()


##### FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE #####
##### FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE #####
##### FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE ########## FOR GPU USE #####

#   --bonus_dir notebooks/bonus_benchmarks/data/bonus \
#   --out_root benchmarks/bonus/outputs_dml05 \
#   --seed_start 2025 \
#   --n_seeds 20 \
#   --n_folds 5 \
#   --n_rep 1 \
#   --use_gpu_mlp

# benchmarks/bonus/05_baseline_doubleml_bonus.py
# ---------------------------------------------------------
# Baseline using DoubleMLIRM with MLP learners on the BONUS data.
# For each seed:
#   - Fit DoubleMLIRM (MLP for outcome and propensity)
#   - Extract out-of-fold nuisance predictions (mu0_hat, mu1_hat, e_hat)
#   - Save them under: out_root/seed_<SEED>/
#   - Compute "paper" WAAE (DR/AIPW-based) for the DML baseline
#
# This script also aggregates WAAE over multiple seeds and writes:
#   out_root/05_dml_multi_summary.json
#
# The per-seed folders are compatible with 07_eval_waae_multi.py
# via: --dml_root <out_root>.
# ---------------------------------------------------------
# import os
# import json
# import time
# import argparse
# from pathlib import Path
#
# import numpy as np
# import doubleml as dml
#
# # sklearn MLP (CPU baseline)
# from sklearn.neural_network import MLPRegressor, MLPClassifier
#
# # Optional: PyTorch + skorch for GPU MLP
# import torch
# import torch.nn as nn
# from skorch import NeuralNetRegressor, NeuralNetClassifier
#
#
# # ---------------------------------------------------------
# # Utilities
# # ---------------------------------------------------------
#
# def find_repo_root(start: Path) -> Path:
#     """Find the git repository root (fallback: a few levels up)."""
#     for p in [start] + list(start.parents):
#         if (p / ".git").exists():
#             return p
#     return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]
#
#
# def load_bonus_arrays(bonus_dir: Path):
#     """Load X, T, Y from BONUS directory."""
#     X = np.load(bonus_dir / "X.npy").astype(np.float32)
#     T = np.load(bonus_dir / "T.npy").astype(np.int64).reshape(-1)
#     Y = np.load(bonus_dir / "Y.npy").astype(np.float32).reshape(-1)
#     return X, T, Y
#
#
# def waae_paper(Y, T, mu0_pred, mu1_pred, e_hat, mu0_anchor, mu1_anchor):
#     r"""
#     WAAE in the style of the r69 paper:
#
#       WAAE(μ̂) = ∑_t | μ̂̄(t) - μ̄(t) | * P_N(t),
#
#     where μ̄(t) is computed via DR/AIPW:
#
#       μ̄(0) = (1/N) * Σ_i [ μ̂0(X_i) + 1{T_i=0}/(1 - ê(X_i)) * (Y_i - μ̂0(X_i)) ]
#       μ̄(1) = (1/N) * Σ_i [ μ̂1(X_i) + 1{T_i=1}/ê(X_i)        * (Y_i - μ̂1(X_i)) ].
#
#     Parameters
#     ----------
#     Y, T : np.ndarray
#         Outcome and treatment indicators.
#     mu0_pred, mu1_pred : np.ndarray
#         Candidate predictions whose WAAE we want to evaluate.
#     e_hat : np.ndarray
#         Propensity score estimates ê(X).
#     mu0_anchor, mu1_anchor : np.ndarray
#         Anchor functions used for the AIPW target means (e.g., DML OOF nuisances).
#
#     Returns
#     -------
#     dict
#         Contains WAAE and intermediate summary statistics.
#     """
#     r1 = float((T == 1).mean())
#     r0 = 1.0 - r1
#
#     mu0_bar_hat = float(mu0_pred.mean())
#     mu1_bar_hat = float(mu1_pred.mean())
#
#     e1 = np.clip(e_hat, 1e-3, 1 - 1e-3)
#     e0 = 1.0 - e1
#     ind0 = (T == 0).astype(np.float32)
#     ind1 = 1.0 - ind0
#
#     aipw0 = mu0_anchor + ind0 / e0 * (Y - mu0_anchor)
#     aipw1 = mu1_anchor + ind1 / e1 * (Y - mu1_anchor)
#     mu0_bar = float(aipw0.mean())
#     mu1_bar = float(aipw1.mean())
#
#     err0 = abs(mu0_bar_hat - mu0_bar)
#     err1 = abs(mu1_bar_hat - mu1_bar)
#     waae = r0 * err0 + r1 * err1
#
#     return {
#         "waae": float(waae),
#         "r0": float(r0),
#         "r1": float(r1),
#         "err0": float(err0),
#         "err1": float(err1),
#         "mu0_mean_pred": float(mu0_bar_hat),
#         "mu1_mean_pred": float(mu1_bar_hat),
#         "mu0_mean_true": float(mu0_bar),
#         "mu1_mean_true": float(mu1_bar),
#     }
#
#
# def _extract_oof(pred_arr: np.ndarray) -> np.ndarray:
#     """
#     Extract an OOF-like prediction vector from DoubleMLIRM.predictions entry.
#
#     In DoubleML, predictions["ml_g0"] / ["ml_g1"] / ["ml_m"] typically have
#     shape (n_obs, n_folds * n_rep) or (n_obs, n_rep, n_folds), depending on version.
#     For robustness, we handle:
#       - 1D: already (n_obs,)
#       - 2D: (n_obs, K) -> average over axis=1
#       - 3D: (n_obs, K1, K2) -> average over axes=(1,2)
#
#     If each observation has exactly one valid cross-fitted prediction per rep,
#     the mean over folds/rep is equivalent to the OOF prediction.
#     """
#     arr = np.asarray(pred_arr)
#     if arr.ndim == 1:
#         return arr.astype(np.float32)
#     if arr.ndim == 2:
#         v = np.nanmean(arr, axis=1)
#         return v.astype(np.float32)
#     if arr.ndim == 3:
#         v = np.nanmean(arr, axis=(1, 2))
#         return v.astype(np.float32)
#     raise ValueError(f"Unexpected prediction array shape: {arr.shape}")
#
#
# # ---------------------------------------------------------
# # MLP builders
# # ---------------------------------------------------------
#
# def build_learners_mlp_sklearn(seed: int):
#     """
#     Build CPU-only sklearn MLP learners for DoubleMLIRM.
#     This is the original baseline version.
#     """
#     ml_g = MLPRegressor(
#         hidden_layer_sizes=(64, 64),
#         activation="relu",
#         solver="adam",
#         alpha=1e-4,
#         learning_rate_init=1e-3,
#         max_iter=500,
#         random_state=seed,
#     )
#     ml_m = MLPClassifier(
#         hidden_layer_sizes=(64, 64),
#         activation="relu",
#         solver="adam",
#         alpha=1e-4,
#         learning_rate_init=1e-3,
#         max_iter=500,
#         random_state=seed + 1000,
#     )
#     return ml_g, ml_m
#
#
# class TorchMLPReg(nn.Module):
#     """
#     Simple regression MLP: input_dim -> hidden -> 1.
#     Used with skorch.NeuralNetRegressor.
#     """
#     def __init__(self, input_dim: int, hidden_sizes=(64, 64)):
#         super().__init__()
#         layers = []
#         d = input_dim
#         for h in hidden_sizes:
#             layers.append(nn.Linear(d, h))
#             layers.append(nn.ReLU())
#             d = h
#         layers.append(nn.Linear(d, 1))
#         net = nn.Sequential(*layers)
#         # Make all parameters double precision to match sklearn/DoubleML (float64)
#         self.net = net.double()
#
#     def forward(self, X):
#         # Ensure input is double; network weights are also double
#         X = X.double()
#         out = self.net(X)
#         return out.squeeze(-1)
#
#
# class TorchMLPClf(nn.Module):
#     """
#     Simple classification MLP: input_dim -> hidden -> 2 logits.
#     Used with skorch.NeuralNetClassifier + CrossEntropyLoss.
#     """
#     def __init__(self, input_dim: int, hidden_sizes=(64, 64), n_classes: int = 2):
#         super().__init__()
#         layers = []
#         d = input_dim
#         for h in hidden_sizes:
#             layers.append(nn.Linear(d, h))
#             layers.append(nn.ReLU())
#             d = h
#         layers.append(nn.Linear(d, n_classes))
#         net = nn.Sequential(*layers)
#         # Same: use double precision
#         self.net = net.double()
#
#     def forward(self, X):
#         # Input as double
#         X = X.double()
#         return self.net(X)
#
#
#
# def build_learners_mlp_torch(seed: int, n_features: int):
#     """
#     Build PyTorch + skorch MLP learners for DoubleMLIRM.
#
#     This version can use GPU (cuda) if available).
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[info] using Torch+skorch MLP on device={device}")
#
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#
#     ml_g = NeuralNetRegressor(
#         module=TorchMLPReg,
#         module__input_dim=n_features,
#         module__hidden_sizes=(64, 64),
#         max_epochs=80,
#         lr=1e-3,
#         batch_size=256,
#         optimizer=torch.optim.Adam,
#         train_split=None,
#         iterator_train__shuffle=True,
#         verbose=0,
#         device=device,
#     )
#
#     ml_m = NeuralNetClassifier(
#         module=TorchMLPClf,
#         module__input_dim=n_features,
#         module__hidden_sizes=(64, 64),
#         max_epochs=80,
#         lr=1e-3,
#         batch_size=256,
#         optimizer=torch.optim.Adam,
#         train_split=None,
#         iterator_train__shuffle=True,
#         verbose=0,
#         device=device,
#         criterion=nn.CrossEntropyLoss,
#     )
#
#     return ml_g, ml_m
#
#
# def build_learners_mlp(seed: int, n_features: int, use_gpu_mlp: bool):
#     """
#     Factory that chooses between sklearn MLP (CPU) and Torch+skorch MLP (GPU).
#
#     Parameters
#     ----------
#     seed : int
#         Random seed.
#     n_features : int
#         Number of covariates (X.shape[1]).
#     use_gpu_mlp : bool
#         If True, use Torch+skorch MLP (GPU if available).
#
#     Returns
#     -------
#     (ml_g, ml_m)
#         Learners for outcome and propensity.
#     """
#     if use_gpu_mlp:
#         return build_learners_mlp_torch(seed, n_features)
#     else:
#         return build_learners_mlp_sklearn(seed)
#
#
# # ---------------------------------------------------------
# # core per-seed experiment
# # ---------------------------------------------------------
#
# def run_one_seed(
#     seed: int,
#     X: np.ndarray,
#     T: np.ndarray,
#     Y: np.ndarray,
#     out_root: Path,
#     n_folds: int = 5,
#     n_rep: int = 1,
#     use_gpu_mlp: bool = False,
# ):
#     """
#     Run DoubleMLIRM for a single seed, save nuisances, and compute WAAE.
#
#     Steps:
#       1. Set up DoubleMLData from X, Y, T.
#       2. Build MLP learners and fit DoubleMLIRM with store_predictions=True.
#       3. Extract OOF nuisances:
#            - mu0_hat(x) (ml_g0)
#            - mu1_hat(x) (ml_g1)
#            - e_hat(x)   (ml_m)
#       4. Save nuisances to out_root/seed_<seed>/.
#       5. Compute paper-style WAAE for DML baseline (using its own nuisances as anchor).
#       6. Save per-seed JSON summary.
#
#     Returns
#     -------
#     float
#         The WAAE value for this seed.
#     """
#     n, p = X.shape
#     dml_data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=T)
#
#     ml_g, ml_m = build_learners_mlp(seed=seed, n_features=p, use_gpu_mlp=use_gpu_mlp)
#     dml_irm = dml.DoubleMLIRM(
#         dml_data,
#         ml_g=ml_g,
#         ml_m=ml_m,
#         n_folds=n_folds,
#         n_rep=n_rep,
#         score="ATE",
#     )
#
#     t0 = time.time()
#     print(f"[seed {seed}] Fitting DoubleMLIRM (MLP, use_gpu_mlp={use_gpu_mlp})…")
#     dml_irm.fit(store_predictions=True)
#     elapsed = time.time() - t0
#     ate = float(dml_irm.coef[0])
#     print(f"[seed {seed}] fit done in {elapsed:.1f}s, ATE={ate:.6f}")
#
#     # Extract nuisance predictions
#     pred_g0 = dml_irm.predictions["ml_g0"]
#     pred_g1 = dml_irm.predictions["ml_g1"]
#     pred_m  = dml_irm.predictions["ml_m"]
#
#     mu0_hat = _extract_oof(pred_g0)
#     mu1_hat = _extract_oof(pred_g1)
#     e_hat   = _extract_oof(pred_m)
#
#     assert len(mu0_hat) == len(X)
#     assert len(mu1_hat) == len(X)
#     assert len(e_hat)   == len(X)
#
#     seed_dir = out_root / f"seed_{seed}"
#     seed_dir.mkdir(parents=True, exist_ok=True)
#
#     np.save(seed_dir / "mu0_hat.npy", mu0_hat.astype(np.float32))
#     np.save(seed_dir / "mu1_hat.npy", mu1_hat.astype(np.float32))
#     np.save(seed_dir / "e_hat.npy",   e_hat.astype(np.float32))
#
#     # Paper-style WAAE for DML-only baseline (OOF nuisances = anchor)
#     waae_stats = waae_paper(
#         Y=Y,
#         T=T,
#         mu0_pred=mu0_hat,
#         mu1_pred=mu1_hat,
#         e_hat=e_hat,
#         mu0_anchor=mu0_hat,
#         mu1_anchor=mu1_hat,
#     )
#
#     ci = dml_irm.confint()
#     ci_low = float(ci.iloc[0, 0])
#     ci_high = float(ci.iloc[0, 1])
#
#     seed_report = {
#         "seed": seed,
#         "ATE": ate,
#         "SE": float(dml_irm.se[0]),
#         "CI95": [ci_low, ci_high],
#         "waae_paper": waae_stats,
#         "timing_sec": float(elapsed),
#         "n_folds": int(n_folds),
#         "n_rep": int(n_rep),
#         "use_gpu_mlp": bool(use_gpu_mlp),
#     }
#
#     with open(seed_dir / "05_dml_seed_summary.json", "w") as f:
#         json.dump(seed_report, f, indent=2)
#
#     print(f"[seed {seed}] WAAE(paper)={waae_stats['waae']:.6f}  -> saved npy & json")
#     return waae_stats["waae"]
#
#
# # ---------------------------------------------------------
# # main
# # ---------------------------------------------------------
#
# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "BONUS baseline using DoubleMLIRM (MLP) across multiple seeds.\n"
#             "For each seed, this script fits DoubleML, saves OOF mu0/mu1/e_hat, "
#             "and computes paper-style WAAE for the DML baseline."
#         )
#     )
#     parser.add_argument(
#         "--bonus_dir",
#         type=str,
#         default="notebooks/bonus_benchmarks/data/bonus",
#         help="Directory containing X.npy, T.npy, Y.npy.",
#     )
#     parser.add_argument(
#         "--out_root",
#         type=str,
#         default="benchmarks/bonus/outputs_dml05",
#         help="Output root under which seed_<seed>/ subfolders will be created.",
#     )
#     parser.add_argument(
#         "--seed_start",
#         type=int,
#         default=2025,
#         help="First seed (e.g., 2025).",
#     )
#     parser.add_argument(
#         "--n_seeds",
#         type=int,
#         default=20,
#         help="Number of seeds to run: seed_start, seed_start+1, ...",
#     )
#     parser.add_argument(
#         "--n_folds",
#         type=int,
#         default=5,
#         help="Number of CV folds for DoubleMLIRM.",
#     )
#     parser.add_argument(
#         "--n_rep",
#         type=int,
#         default=1,
#         help="Number of DoubleML repetitions.",
#     )
#     parser.add_argument(
#         "--use_gpu_mlp",
#         action="store_true",
#         help="If set, use PyTorch+skorch MLP (GPU if available) instead of sklearn MLP.",
#     )
#
#     args = parser.parse_args()
#
#     here = Path(__file__).resolve()
#     repo_root = find_repo_root(here)
#     bonus_dir = (repo_root / args.bonus_dir).resolve()
#     out_root = (repo_root / args.out_root).resolve()
#     out_root.mkdir(parents=True, exist_ok=True)
#
#     print(f"[info] repo_root   = {repo_root}")
#     print(f"[info] BONUS_DIR   = {bonus_dir}")
#     print(f"[info] OUT_ROOT    = {out_root}")
#     print(f"[info] use_gpu_mlp = {args.use_gpu_mlp}")
#
#     X, T, Y = load_bonus_arrays(bonus_dir)
#     n, p = X.shape
#     print(f"[info] X={X.shape}, T={T.shape}, Y={Y.shape}")
#
#     seeds = [args.seed_start + i for i in range(args.n_seeds)]
#     waae_list = []
#
#     t0_all = time.time()
#     for s in seeds:
#         waae_s = run_one_seed(
#             seed=s,
#             X=X,
#             T=T,
#             Y=Y,
#             out_root=out_root,
#             n_folds=args.n_folds,
#             n_rep=args.n_rep,
#             use_gpu_mlp=args.use_gpu_mlp,
#         )
#         waae_list.append(waae_s)
#
#     elapsed_all = time.time() - t0_all
#     waae_arr = np.asarray(waae_list, dtype=float)
#
#     agg = {
#         "seeds": seeds,
#         "waae_per_seed": [float(w) for w in waae_arr],
#         "waae_mean": float(waae_arr.mean()),
#         "waae_std": float(waae_arr.std(ddof=1)) if len(waae_arr) > 1 else 0.0,
#         "elapsed_total_sec": float(elapsed_all),
#         "n": int(n),
#         "p": int(p),
#         "n_folds": int(args.n_folds),
#         "n_rep": int(args.n_rep),
#         "use_gpu_mlp": bool(args.use_gpu_mlp),
#     }
#
#     summary_path = out_root / "05_dml_multi_summary.json"
#     with open(summary_path, "w") as f:
#         json.dump(agg, f, indent=2)
#
#     print("\n[SUMMARY over seeds]")
#     print(f"  seeds          : {seeds[0]} .. {seeds[-1]} (total {len(seeds)})")
#     print(f"  WAAE mean      : {agg['waae_mean']:.6f}")
#     print(f"  WAAE std       : {agg['waae_std']:.6f}")
#     print(f"  total time (s) : {agg['elapsed_total_sec']:.1f}")
#     print(f"[OK] saved -> {summary_path}")
#
#
# if __name__ == "__main__":
#     main()
