import argparse
from pathlib import Path
import math
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# Your project-specific imports (kept from your original script)
from datasets.rearrange_dset import load_rearrange_slice_train_val
from datasets.img_transforms import default_transform
from models.visual_world_model import VWorldModel

from omegaconf import OmegaConf
from plan import load_model

# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 42):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Data / Model builders
# -----------------------------

def build_world_model(
    ckpt_path: str,
    img_size: int,
    num_hist: int,
    num_pred: int,
    device: torch.device,
) -> VWorldModel:
    """Load a ``VWorldModel`` from a training checkpoint.

    This utility expects ``ckpt_path`` to point to a file inside a training
    output directory produced by ``train.py``.  The corresponding Hydra
    configuration (``hydra.yaml``) is loaded to instantiate the model with the
    correct encoder/decoder components before restoring the checkpointed
    weights.

    The ``img_size``, ``num_hist`` and ``num_pred`` arguments are kept for
    backward compatibility but override the values in the loaded config if
    provided."""

    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.parent.parent / "hydra.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg.img_size = img_size
    cfg.num_hist = num_hist
    cfg.num_pred = num_pred

    wm = load_model(ckpt_path, cfg, getattr(cfg, "num_action_repeat", 1), device=device)
    wm.eval()  # we only encode with it
    return wm


def prepare_dataloaders(batch_size: int, data_path: str, n_rollout: int, num_workers: int) -> Tuple[DataLoader, DataLoader, int]:
    train_set, val_set, num_actions = load_rearrange_slice_train_val(
        data_root=data_path,
        transform=default_transform,
        n_rollout=n_rollout,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, num_actions


# -----------------------------
# Inverse Projector Analysis
# -----------------------------

def collect_inverse_latents(
    wm: VWorldModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect inverse-projector latents and the corresponding action labels.
    Returns:
        latents: (N, D)
        labels:  (N,)
    """
    latents_list, labels_list = [], []
    wm.eval()
    with torch.no_grad():
        for b_idx, (obs, act, _) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            obs = {k: v.to(device) for k, v in obs.items()}
            act = act.to(device).squeeze(-1).long()  # (B,T)
            z_obs = wm.encode_obs(obs)
            latent_act, _ = wm.encode_act(None, z_obs["visual_frame"])  # (B,T,1,D) or similar
            latent_act = latent_act.squeeze(2)[:, 1:, :]  # align with labels; (B,T-1,D)
            labels = act[:, :-1]  # (B,T-1)

            latents_list.append(latent_act.reshape(-1, latent_act.shape[-1]).cpu())
            labels_list.append(labels.reshape(-1).cpu())

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    # Filter any padding labels (<0) if present
    valid = labels >= 0
    return latents[valid], labels[valid]


def gaussian_from_latents(latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, sigma) where sigma is the full covariance matrix.
    For planners that prefer a diagonal covariance, you can take diag(sigma).
    """
    x = latents.double()
    mu = x.mean(dim=0)
    xc = x - mu
    # Unbiased covariance (N-1 in denom)
    n = x.shape[0]
    sigma = (xc.t() @ xc) / max(n - 1, 1)
    return mu.float(), sigma.float()


def per_action_stats(latents: torch.Tensor, labels: torch.Tensor, num_actions: int) -> Dict[str, torch.Tensor]:
    dim = latents.shape[-1]
    mus = torch.zeros(num_actions, dim)
    covs = torch.zeros(num_actions, dim, dim)
    counts = torch.zeros(num_actions, dtype=torch.long)

    for a in range(num_actions):
        idx = (labels == a)
        if idx.any():
            mu_a, cov_a = gaussian_from_latents(latents[idx])
            mus[a] = mu_a
            covs[a] = cov_a
            counts[a] = idx.sum()
        else:
            mus[a] = torch.nan
            covs[a] = torch.nan
    return {"mus": mus, "covs": covs, "counts": counts}


def cosine_similarity_matrix(centroids: torch.Tensor) -> torch.Tensor:
    # centroids: (K,D)
    c = centroids
    c = c / (c.norm(dim=1, keepdim=True) + 1e-9)
    return c @ c.t()  # (K,K)


def nearest_centroid_accuracy(latents: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor) -> float:
    # latents: (N,D), labels: (N,), centroids: (K,D)
    with torch.no_grad():
        # Cosine distance to centroids
        c_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-9)
        x_norm = latents / (latents.norm(dim=1, keepdim=True) + 1e-9)
        sims = x_norm @ c_norm.t()  # (N,K)
        preds = sims.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
    return float(correct) / max(total, 1)


def quality_report(
    wm: VWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_actions: int,
    max_batches: int = None,
) -> Dict:
    """Compute a set of projector quality metrics and return a dict of tensors/values.
    - Global Gaussian (mu, Sigma)
    - Per-action (mu_a, Sigma_a)
    - Centroid cosine similarity matrix
    - Nearest-centroid accuracy (train/val)
    - Intra-class vs inter-class cosine stats
    """
    lat_tr, y_tr = collect_inverse_latents(wm, train_loader, device, max_batches)
    lat_va, y_va = collect_inverse_latents(wm, val_loader, device, max_batches)

    mu_g, cov_g = gaussian_from_latents(lat_tr)
    per_tr = per_action_stats(lat_tr, y_tr, num_actions)

    centroids = per_tr["mus"].clone()
    # Replace NaN centroids (classes unseen in train) with zeros to avoid crashes
    nan_mask = torch.isnan(centroids).any(dim=1)
    centroids[nan_mask] = 0.0

    sim_mat = cosine_similarity_matrix(centroids)

    nc_acc_tr = nearest_centroid_accuracy(lat_tr, y_tr, centroids)
    nc_acc_va = nearest_centroid_accuracy(lat_va, y_va, centroids)

    # Intra-/Inter-class cosine: sample pairs efficiently via centroids distances as proxy
    # (For a more exact measure, compute pairwise within-class cosine on a subsample)
    with torch.no_grad():
        # Approximate intra as average cosine between samples and their centroid
        x_norm = lat_tr / (lat_tr.norm(dim=1, keepdim=True) + 1e-9)
        c_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-9)
        cos_to_centroid = []
        for a in range(num_actions):
            idx = (y_tr == a)
            if idx.any():
                cos_vals = (x_norm[idx] * c_norm[a]).sum(dim=1)
                cos_to_centroid.append(cos_vals.mean().item())
            else:
                cos_to_centroid.append(float("nan"))
        avg_intra_proxy = np.nanmean(cos_to_centroid)

        # Inter-class proxy: average off-diagonal centroid cosine
        with torch.no_grad():
            offdiag = sim_mat.cpu().numpy()
            mask = ~np.eye(num_actions, dtype=bool)
            avg_inter_proxy = offdiag[mask].mean() if mask.sum() > 0 else float("nan")

    report = {
        "global_mu": mu_g,
        "global_cov": cov_g,
        "per_action_mus": per_tr["mus"],
        "per_action_covs": per_tr["covs"],
        "per_action_counts": per_tr["counts"],
        "centroid_cosine": sim_mat,
        "nearest_centroid_acc_train": nc_acc_tr,
        "nearest_centroid_acc_val": nc_acc_va,
        "avg_intra_cosine_proxy": float(avg_intra_proxy),
        "avg_inter_cosine_proxy": float(avg_inter_proxy),
    }
    return report


def save_quality_report(report: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save torch tensors as npy for easy downstream use
    np.save(out_dir / "global_mu.npy", report["global_mu"].cpu().numpy())
    np.save(out_dir / "global_cov.npy", report["global_cov"].cpu().numpy())
    np.save(out_dir / "per_action_mus.npy", report["per_action_mus"].cpu().numpy())
    np.save(out_dir / "per_action_covs.npy", report["per_action_covs"].cpu().numpy())
    np.save(out_dir / "per_action_counts.npy", report["per_action_counts"].cpu().numpy())
    np.save(out_dir / "centroid_cosine.npy", report["centroid_cosine"].cpu().numpy())

    # Compact summary JSON-like (npz)
    np.savez_compressed(
        out_dir / "projector_stats.npz",
        global_mu=report["global_mu"].cpu().numpy(),
        global_cov=report["global_cov"].cpu().numpy(),
        per_action_mus=report["per_action_mus"].cpu().numpy(),
        per_action_covs=report["per_action_covs"].cpu().numpy(),
        per_action_counts=report["per_action_counts"].cpu().numpy(),
        centroid_cosine=report["centroid_cosine"].cpu().numpy(),
        nc_acc_train=report["nearest_centroid_acc_train"],
        nc_acc_val=report["nearest_centroid_acc_val"],
        avg_intra_cosine_proxy=report["avg_intra_cosine_proxy"],
        avg_inter_cosine_proxy=report["avg_inter_cosine_proxy"],
    )


# -----------------------------
# Classifier Training
# -----------------------------

def build_classifier(input_dim: int, num_actions: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_actions),
    )


def run_epoch(
    model: nn.Module,
    wm: VWorldModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    grad_clip: float = 0.0,
    log_every: int = 50,
    use_tqdm: bool = True,
) -> Tuple[float, int]:
    model.train()
    running_loss = 0.0
    total_steps = len(loader)
    iterator = enumerate(loader, start=1)
    if use_tqdm and TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=total_steps, desc="train", leave=False)

    for step, (obs, act, _) in iterator:
        obs = {k: v.to(device) for k, v in obs.items()}
        act = act.to(device).squeeze(-1).long()  # (B,T)

        with torch.no_grad():
            z_obs = wm.encode_obs(obs)
            latent_act, _ = wm.encode_act(None, z_obs["visual_frame"])  # (B,T,1,D)
        latent_act = latent_act.squeeze(2)[:, 1:, :]  # (B,T-1,D)
        labels = act[:, :-1]

        logits = model(latent_act.reshape(-1, latent_act.shape[-1]))
        loss = criterion(logits, labels.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        if not use_tqdm or not TQDM_AVAILABLE:
            # Simple periodic logging with steps left
            if step % log_every == 0 or step == total_steps:
                steps_left = total_steps - step
                avg_loss = running_loss / min(log_every, step)
                print(f"[train] step {step}/{total_steps} | steps_left={steps_left} | loss={avg_loss:.4f}")
                running_loss = 0.0
        else:
            iterator.set_postfix({"loss": f"{loss.item():.4f}", "steps_left": total_steps - step})

    # Average loss over all steps (re-compute robustly)
    return float(running_loss) / max(1, (total_steps % log_every or log_every)), total_steps


def evaluate(
    model: nn.Module,
    wm: VWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct, total = 0, 0
    iterator = enumerate(loader, start=1)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=len(loader), desc="valid", leave=False)

    with torch.no_grad():
        for _, (obs, act, _) in iterator:
            obs = {k: v.to(device) for k, v in obs.items()}
            act = act.to(device).squeeze(-1).long()
            z_obs = wm.encode_obs(obs)
            latent_act, _ = wm.encode_act(None, z_obs["visual_frame"])  # (B,T,1,D)
            latent_act = latent_act.squeeze(2)[:, 1:, :]
            labels = act[:, :-1]

            logits = model(latent_act.reshape(-1, latent_act.shape[-1]))
            preds = logits.argmax(dim=-1)
            correct += (preds == labels.reshape(-1)).sum().item()
            total += preds.numel()
    return (correct / total) if total > 0 else 0.0


# -----------------------------
# Main training entry
# -----------------------------

def train_classifier(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # World model (inverse projector comes from wm.encode_act)
    wm = build_world_model(args.ckpt,args.image_size,args.num_hist, args.num_pred, device)

    # Data
    train_loader, val_loader, num_actions = prepare_dataloaders(
        args.batch_size, args.data_path, args.n_rollout, args.num_workers
    )

    # Peek latent dim by a single forward pass
    with torch.no_grad():
        obs0, act0, _ = next(iter(train_loader))
        obs0 = {k: v.to(device) for k, v in obs0.items()}
        z0 = wm.encode_obs(obs0)
        l0, _ = wm.encode_act(None, z0["visual_frame"])  # (B,T,1,D)
        latent_dim = int(l0.shape[-1])
    print(f"[info] latent_dim={latent_dim}, num_actions={num_actions}")

    # Classifier
    classifier = build_classifier(latent_dim, num_actions, hidden=args.hidden).to(device)
    print(f"[info] classifier params: {count_parameters(classifier):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine_decay:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 1
    best_val = -1.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        classifier.load_state_dict(ckpt["classifier_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", -1.0))
        print(f"[resume] from {args.resume} at epoch {start_epoch} (best_val={best_val:.4f})")

    # Training loop
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss, steps = run_epoch(
            classifier, wm, train_loader, device, criterion, optimizer,
            grad_clip=args.grad_clip, log_every=args.log_every, use_tqdm=args.tqdm
        )
        val_acc = evaluate(classifier, wm, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:03d}] loss={train_loss:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.6f} | {epoch_time:.1f}s")

        # Save checkpoint
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"classifier_epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "classifier_state": classifier.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "best_val": max(best_val, val_acc),
                "latent_dim": latent_dim,
                "num_actions": num_actions,
            },
            ckpt_path,
        )
        print(f"[save] {ckpt_path}")

        # Optional: projector analysis every N epochs (or at the end)
        if args.analyze_every > 0 and (epoch % args.analyze_every == 0 or epoch == args.epochs):
            print("[analysis] computing inverse projector quality metrics...")
            report = quality_report(
                wm, train_loader, val_loader, device, num_actions, max_batches=args.analysis_max_batches
            )
            qa_dir = save_dir / f"analysis_epoch_{epoch}"
            save_quality_report(report, qa_dir)
            print(
                f"[analysis] saved stats to {qa_dir} | "
                f"nc_train={report['nearest_centroid_acc_train']:.4f} | "
                f"nc_val={report['nearest_centroid_acc_val']:.4f} | "
                f"intra≈{report['avg_intra_cosine_proxy']:.3f} | inter≈{report['avg_inter_cosine_proxy']:.3f}"
            )


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action classifier on inverse-projector latents + analyze projector quality")
    # Paths & data
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained VWorldModel checkpoint")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./classifier_ckpts")

    # Model / training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--cosine_decay", action="store_true")

    # World model encoder config
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_hist", type=int, default=3)
    parser.add_argument("--num_pred", type=int, default=1)

    # Data specifics
    parser.add_argument("--n_rollout", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    # Logging
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm progress bars")

    # Analysis
    parser.add_argument("--analyze_every", type=int, default=1, help="Run projector analysis every N epochs (0 to disable)")
    parser.add_argument("--analysis_max_batches", type=int, default=None, help="Cap batches when collecting latents for faster analysis")

    # Misc
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_classifier(args)
