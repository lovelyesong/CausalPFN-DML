# benchmarks/bonus/06_train_pfn.py
# ---------------------------------------------------------------
# PFN-lite (MLP) training skeleton on prior shards, with optional
# (B) DML-guided penalty and (C) shift weights from domain classifier.

# (A) PFN
# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_A
# --epochs 5 --batch_size 1024 --seed 2025

# (B)  (PFN + DML penalty, shift 없음)
# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_B
# --use_penalty --lambda_penalty 1.0
# --epochs 5 --batch_size 1024 --seed 2025

# (B)  (PFN + DML penalty, shift 없음) - C의 lambda 30과 동일
# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_Bx1p44
# --use_penalty
# --lambda_penalty 1.44
# --epochs 5 --batch_size 1024 --seed 2025

# (c) (PFN + DML penalty + shift 가중)
# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_C
# --use_penalty --use_shift --lambda_penalty 1.0
# --epochs 5 --batch_size 1024 --seed 2025

# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_Cx30
# --use_penalty --use_shift --lambda_penalty 30.0
# --epochs 5 --batch_size 1024 --seed 2025

# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_Cx21
# --use_penalty --use_shift --lambda_penalty 21.0
# --epochs 10 --batch_size 1024 --seed 2025

# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_Cx24
# --use_penalty --use_shift --lambda_penalty 24.0
# --epochs 10 --batch_size 1024 --seed 2025



#
# What this script does:
# - Streams prior shards (X, T, Y, mu0_true, mu1_true, e_true) from data/prior/...
# - Trains a simple MLP to predict tau_true = mu1_true - mu0_true from X
# - (Optional) Adds a penalty towards DR/DML CATE tau_hat on Bonus data
# - (Optional) Uses per-sample shift weight zeta = P(bonus|x) to modulate penalty
# - Saves a checkpoint and a JSON training summary
#
# Replace the MLP with your actual CausalPFN model later; keep the API
#    tau_pred = model(X)
# ---------------------------------------------------------------

import os
import glob
import json
import math
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------- utils -----------------------------
def find_repo_root(start: Path) -> Path:
    """Walk upwards to find the repo root by locating a .git directory."""
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start.parents[3] if len(start.parents) >= 4 else start.parents[-1]


def load_npz_shard(path: str) -> Dict[str, np.ndarray]:
    """Load one prior shard as a dict of numpy arrays."""
    d = np.load(path)
    # Defensive: older shards may or may not have tau_true stored; reconstruct if needed
    out = {k: d[k] for k in d.files}
    if "tau_true" not in out and all(k in out for k in ("mu0_true", "mu1_true")):
        out["tau_true"] = (out["mu1_true"] - out["mu0_true"]).astype(np.float32)
    return out


def iter_prior_shards(prior_dir: Path, shuffle_files: bool = True, seed: int = 42) -> Iterator[Dict[str, np.ndarray]]:
    """Yield dicts for shard_*.npz from the prior directory."""
    paths = sorted(glob.glob(str(prior_dir / "shard_*.npz")))
    if shuffle_files:
        rng = np.random.default_rng(seed)
        rng.shuffle(paths)
    for p in paths:
        yield load_npz_shard(p)


def batch_iterator_from_array(X: np.ndarray,
                              y: np.ndarray,
                              batch_size: int,
                              shuffle: bool = True,
                              seed: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple numpy mini-batch iterator.
    - X: (n, p) features
    - y: (n,) targets
    """
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for s in range(0, n, batch_size):
        j = idx[s:s + batch_size]
        yield X[j], y[j]


def to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Numpy array -> torch tensor on device."""
    return torch.tensor(x, dtype=dtype, device=device)


# ----------------------------- model -----------------------------
class PFNLite(nn.Module):
    """
    Minimal PFN-like head: an MLP mapping X -> tau(x).
    Replace this with your actual CausalPFN module later; keep .forward(X)->tau_pred.
    """
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns tau_pred of shape (batch, 1)
        return self.net(x)


# --------------------------- training ----------------------------
def train_one_epoch(model: nn.Module,
                    optimizers: Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]],
                    device: torch.device,
                    prior_batches: Iterator[Tuple[np.ndarray, np.ndarray]],
                    bonus_payload: Optional[Dict[str, np.ndarray]],
                    lambda_penalty: float,
                    use_penalty: bool,
                    use_shift: bool,
                    penalty_batch_size: int = 512) -> Dict[str, float]:
    """
    One training epoch over prior batches, optionally mixing in a penalty step from Bonus.
    - prior_batches: iterable of (X_prior, tau_true) numpy mini-batches
    - bonus_payload: dict with X_bonus, tau_hat (DR), zeta (shift weights)
    - use_penalty: if True, add penalty towards tau_hat on Bonus
    - use_shift: if True, weight the penalty by zeta (else uniform weight)
    """
    optimizer, scheduler = optimizers
    model.train()
    mse = nn.MSELoss(reduction="mean")

    # Pre-load bonus arrays to device if penalty is on
    if use_penalty and bonus_payload is not None:
        Xb = bonus_payload["X_bonus"]
        th = bonus_payload["tau_hat"]
        zb = bonus_payload["zeta"] if use_shift else np.ones_like(th, dtype=np.float32)
        # sanity: clip weights in [0,1]
        zb = np.clip(zb, 0.0, 1.0).astype(np.float32)
    else:
        Xb = th = zb = None

    n_prior_seen, n_penalty_seen = 0, 0
    loss_prior_accum, loss_pen_accum = 0.0, 0.0
    steps_prior, steps_penalty = 0, 0

    for X_np, tau_np in prior_batches:
        # ----- (A) prior step: fit tau_true on synthetic shards -----
        optimizer.zero_grad(set_to_none=True)

        X_t = to_tensor(X_np, device)
        y_t = to_tensor(tau_np.reshape(-1, 1), device)  # (B,1)
        y_hat = model(X_t)
        loss_prior = mse(y_hat, y_t)
        loss = loss_prior

        # ----- (B) optional Bonus penalty towards DR/DML tau_hat -----
        if use_penalty and Xb is not None:
            # Sample a random penalty mini-batch from Bonus
            nB = Xb.shape[0]
            idx = np.random.randint(0, nB, size=min(penalty_batch_size, nB))
            Xp = to_tensor(Xb[idx], device)
            th_p = to_tensor(th[idx].reshape(-1, 1), device)
            wz = to_tensor(zb[idx].reshape(-1, 1), device)
            # Weighted L2 penalty: lambda * w * || tau_pred - tau_hat ||^2
            tau_pred_bonus = model(Xp)
            pen = ((tau_pred_bonus - th_p) ** 2) * wz
            loss_penalty = lambda_penalty * pen.mean()
            loss = loss + loss_penalty

            loss_pen_accum += float(loss_penalty.detach().cpu().item())
            n_penalty_seen += len(idx)
            steps_penalty += 1

        # ----- backward / step -----
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_prior_accum += float(loss_prior.detach().cpu().item())
        n_prior_seen += X_np.shape[0]
        steps_prior += 1

    stats = dict(
        loss_prior=loss_prior_accum / max(1, steps_prior),
        loss_penalty=(loss_pen_accum / max(1, steps_penalty)) if use_penalty else 0.0,
        n_prior=n_prior_seen,
        n_bonus_penalty=n_penalty_seen,
        steps_prior=steps_prior,
        steps_penalty=steps_penalty,
    )
    return stats


# --------------------------- main entry ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PFN-lite on prior shards with optional DML penalty & shift weights.")
    # Paths
    parser.add_argument("--prior_dir", type=str, default="data/prior/bonus", help="Folder with shard_*.npz (prior).")
    parser.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus", help="Folder with X.npy/T.npy/Y.npy.")
    parser.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs", help="Where to save checkpoints & logs.")
    # Optional guidance artifacts
    parser.add_argument("--tau_hat_path", type=str, default="benchmarks/bonus/outputs/tau_hat.npy", help="DR/DML CATE per sample.")
    parser.add_argument("--zeta_path", type=str, default="benchmarks/bonus/outputs/05_zeta_bonus.npy", help="Shift score per sample.")
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--penalty_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Switches
    parser.add_argument("--use_penalty", action="store_true", help="Enable DML-guided penalty on Bonus.")
    parser.add_argument("--use_shift", action="store_true", help="Weight penalty by shift score zeta.")
    parser.add_argument("--lambda_penalty", type=float, default=1.0, help="Penalty multiplier.")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    # Resolve repo-rooted paths
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)
    prior_dir = (repo_root / args.prior_dir).resolve()
    bonus_dir = (repo_root / args.bonus_dir).resolve()
    outputs_dir = (repo_root / args.outputs_dir).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load one shard to infer input dimension
    first_shard = next(iter_prior_shards(prior_dir, shuffle_files=False))
    p = int(first_shard["X"].shape[1])

    # Build model & optimizer
    model = PFNLite(in_dim=p, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None  # Optionally add cosine/step LR later

    # Prepare Bonus payload if penalty is enabled
    bonus_payload = None
    if args.use_penalty:
        X_bonus = np.load(bonus_dir / "X.npy").astype(np.float32)
        tau_hat = np.load(repo_root / args.tau_hat_path).astype(np.float32)
        assert len(tau_hat.shape) == 1, "tau_hat.npy should be shape (n_bonus,)"
        assert X_bonus.shape[0] == tau_hat.shape[0], "X and tau_hat size mismatch."

        if args.use_shift:
            zeta_path = repo_root / args.zeta_path
            if not zeta_path.exists():
                raise FileNotFoundError(f"Shift file not found: {zeta_path}")
            zeta = np.load(zeta_path).astype(np.float32)
            assert zeta.shape[0] == X_bonus.shape[0], "zeta and X_bonus size mismatch."
        else:
            zeta = np.ones_like(tau_hat, dtype=np.float32)

        bonus_payload = {"X_bonus": X_bonus, "tau_hat": tau_hat, "zeta": zeta}

    # Recreate shard iterator (first shard got consumed)
    def prior_epoch_iter(seed_offset: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield mini-batches (X, tau_true) across all shards for one epoch."""
        gen = iter_prior_shards(prior_dir, shuffle_files=True, seed=args.seed + seed_offset)
        for shard in gen:
            Xs = shard["X"].astype(np.float32)
            taus = shard["tau_true"].astype(np.float32)
            # Iterate batches within shard
            for xb, yb in batch_iterator_from_array(Xs, taus, batch_size=args.batch_size,
                                                    shuffle=True, seed=args.seed + seed_offset):
                yield xb, yb

    # Training loop
    history: List[Dict[str, Any]] = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model=model,
            optimizers=(optimizer, scheduler),
            device=device,
            prior_batches=prior_epoch_iter(seed_offset=epoch),
            bonus_payload=bonus_payload,
            lambda_penalty=args.lambda_penalty,
            use_penalty=args.use_penalty,
            use_shift=args.use_shift,
            penalty_batch_size=args.penalty_batch_size,
        )
        elapsed = time.time() - t_start
        log = dict(epoch=epoch, elapsed_sec=round(elapsed, 2), **stats)
        history.append(log)
        print(f"[epoch {epoch}/{args.epochs}] "
              f"loss_prior={log['loss_prior']:.6f} "
              f"loss_penalty={log['loss_penalty']:.6f} "
              f"n_prior={log['n_prior']} n_bonus_penalty={log['n_bonus_penalty']}")

    # Save checkpoint & logs
    ckpt_path = outputs_dir / "06_pfnlite.ckpt"
    torch.save({"model": model.state_dict(),
                "cfg": vars(args),
                "p": p}, ckpt_path)
    with open(outputs_dir / "06_train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[OK] Saved checkpoint -> {ckpt_path}")
    print(f"[OK] Saved log -> {outputs_dir / '06_train_log.json'}")


if __name__ == "__main__":
    main()
