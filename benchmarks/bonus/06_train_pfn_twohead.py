# benchmarks/bonus/06_train_pfn_twohead.py
# --------------------------------------------------------------------
# Train a "PFN-lite Two-Head" model that directly predicts (mu0(x), mu1(x))
# from prior-simulation shards (NPZ files). No DML/DR penalty or shift is
# used here: this is the clean PFN-only prior training for the A-baseline.
#
# Expected prior directory layout:
#   prior_dir/
#     shard_0000.npz, shard_0001.npz, ...
# Each shard contains arrays with the same length n:
#   X, mu0_true, mu1_true    (and possibly tau_true, e_true, T, Y)
#
# It saves:
#   outputs_dir/06_pfn_twohead.ckpt   (PyTorch checkpoint)
#     └─ includes PFN-only predictions on BONUS:
#        mu0_pfn, mu1_pfn, tau_pfn (over BONUS X)
#   outputs_dir/06_train_log_twohead.json
# --------------------------------------------------------------------

# --prior_dir data/prior/bonus
# --bonus_dir notebooks/bonus_benchmarks/data/bonus
# --outputs_dir benchmarks/bonus/outputs_A_twohead
# --epochs 10 --steps_per_epoch 400
# --batch_size 1024 --lr 1e-3 --seed 2025

import os
import json
import time
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------- Model ----------------------------

class PFNLiteTwoHead(nn.Module):
    """
    Simple MLP with two-headed output: [mu0(x), mu1(x)].
    """
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
            d = hidden
        layers += [nn.Linear(d, 2)]  # output: [mu0, mu1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns (n, 2): [:,0]=mu0_hat, [:,1]=mu1_hat
        return self.net(x)


# ------------------------- Data utilities -------------------------

def list_shards(prior_dir: Path):
    files = sorted(glob.glob(str(prior_dir / "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"No shard_*.npz found under {prior_dir}")
    return files


def load_random_batch(shard_paths, batch_size, rng: np.random.Generator):
    """
    Pick a random shard, then pick a random batch of rows out of it.
    Returns (X_batch, mu0_batch, mu1_batch) as float32 numpy arrays.
    """
    shard_path = shard_paths[rng.integers(0, len(shard_paths))]
    with np.load(shard_path) as Z:
        X = Z["X"].astype(np.float32)
        mu0 = Z["mu0_true"].astype(np.float32)
        mu1 = Z["mu1_true"].astype(np.float32)

    n = X.shape[0]
    idx = rng.integers(0, n, size=batch_size)
    return X[idx], mu0[idx], mu1[idx]


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train PFN-lite Two-Head (mu0, mu1) from prior shards.")
    ap.add_argument("--prior_dir", type=str, default="data/prior/bonus",
                    help="Folder containing shard_*.npz prior simulations.")
    ap.add_argument("--outputs_dir", type=str, default="benchmarks/bonus/outputs_A_twohead",
                    help="Where to write ckpt/logs.")
    ap.add_argument("--bonus_dir", type=str, default="notebooks/bonus_benchmarks/data/bonus",
                    help="Folder with BONUS X.npy (for exporting mu0_pfn/mu1_pfn on BONUS).")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps_per_epoch", type=int, default=400,
                    help="Num of mini-batch updates per epoch (sampled with replacement).")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_dir = Path(args.prior_dir).resolve()
    out_dir = Path(args.outputs_dir).resolve()
    bonus_dir = Path(args.bonus_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shard_paths = list_shards(prior_dir)

    # Probe input dimension from one shard
    with np.load(shard_paths[0]) as Z0:
        p = int(Z0["X"].shape[1])

    model = PFNLiteTwoHead(in_dim=p, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    log = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for _ in range(args.steps_per_epoch):
            Xb, mu0b, mu1b = load_random_batch(shard_paths, args.batch_size, rng)

            Xt = torch.tensor(Xb, dtype=torch.float32, device=device)
            mu0t = torch.tensor(mu0b, dtype=torch.float32, device=device).unsqueeze(1)
            mu1t = torch.tensor(mu1b, dtype=torch.float32, device=device).unsqueeze(1)

            opt.zero_grad(set_to_none=True)
            pred = model(Xt)                 # (bs, 2)
            mu0_pred = pred[:, :1]
            mu1_pred = pred[:, 1:]

            # MSE over both heads
            loss = loss_fn(mu0_pred, mu0t) + loss_fn(mu1_pred, mu1t)
            loss.backward()
            opt.step()

            running += float(loss.item())

        elapsed = time.time() - t0
        avg_loss = running / args.steps_per_epoch
        entry = {
            "epoch": epoch,
            "elapsed_sec": elapsed,
            "loss_mse": avg_loss,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size
        }
        log.append(entry)
        print(f"[epoch {epoch:03d}] loss_mse={avg_loss:.4f} elapsed={elapsed:.1f}s")

    # ---------------- Inference on BONUS X (for PFN-only WAAE) ----------------
    bonus_X_path = bonus_dir / "X.npy"
    if not bonus_X_path.exists():
        raise FileNotFoundError(f"BONUS X not found at {bonus_X_path} (set --bonus_dir correctly).")
    X_bonus = np.load(bonus_X_path).astype(np.float32)
    if X_bonus.shape[1] != p:
        raise ValueError(f"BONUS X has p={X_bonus.shape[1]} but model was trained with p={p}.")

    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_bonus).to(device)
        pred = model(Xt)  # (n,2)
        if pred.ndim != 2 or pred.shape[1] != 2:
            raise RuntimeError("Two-head model must output shape (n,2).")
        mu0_bonus = pred[:, 0].detach().cpu().numpy().astype(np.float32)
        mu1_bonus = pred[:, 1].detach().cpu().numpy().astype(np.float32)
        tau_bonus = (mu1_bonus - mu0_bonus).astype(np.float32)

    # ---------------- Save checkpoint (includes PFN-only μ over BONUS) ----------------
    ckpt_path = out_dir / "06_pfn_twohead.ckpt"
    cfg = dict(
        hidden=args.hidden, depth=args.depth, dropout=args.dropout,
        two_head=True,  # IMPORTANT FLAG
    )
    torch.save(
        {
            "cfg": cfg,
            "p": p,
            "model": model.state_dict(),
            # PFN-only predictions on BONUS (for 07_eval_waae --use_ckpt_mu)
            "mu0_pfn": torch.from_numpy(mu0_bonus),
            "mu1_pfn": torch.from_numpy(mu1_bonus),
            "tau_pfn": torch.from_numpy(tau_bonus),
            "meta": {
                "bonus_dir": str(bonus_dir),
                "trained_from": str(prior_dir),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        },
        ckpt_path,
    )
    print(f"[OK] saved ckpt -> {ckpt_path}")

    # Save log
    with open(out_dir / "06_train_log_twohead.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[OK] saved log -> {out_dir / '06_train_log_twohead.json'}")


if __name__ == "__main__":
    main()
