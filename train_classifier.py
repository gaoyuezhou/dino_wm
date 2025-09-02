import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf


from datasets.rearrange_dset import load_rearrange_slice_train_val
from datasets.img_transforms import default_transform
from models.visual_world_model import VWorldModel


def build_world_model(ckpt_path: str, img_size: int, num_hist: int, num_pred: int):

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = None
    hydra_cfg_path = ckpt_path.parent.parent / "hydra.yaml"
    if ("encoder" not in ckpt or "action_encoder" not in ckpt or "proprio_encoder" not in ckpt) and hydra_cfg_path.exists():
        cfg = OmegaConf.load(hydra_cfg_path)

    encoder = ckpt.get("encoder")
    if encoder is None:
        if cfg is None:
            raise KeyError("encoder not found in checkpoint and no hydra config available")
        encoder = hydra.utils.instantiate(cfg.encoder)

    proprio_encoder = ckpt.get("proprio_encoder")
    if proprio_encoder is None:
        if cfg is None:
            raise KeyError("proprio_encoder missing and no hydra config available")
        proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder)

    action_encoder = ckpt.get("action_encoder")
    if action_encoder is None:
        if cfg is None:
            raise KeyError("action_encoder missing and no hydra config available")
        action_encoder = hydra.utils.instantiate(cfg.action_encoder)
    wm = VWorldModel(
        image_size=img_size,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        proprio_encoder=ckpt["proprio_encoder"],
        action_encoder=ckpt["action_encoder"],
        decoder=None,
        predictor=None,
        proprio_dim=0,
        action_dim=0,
        concat_dim=1,
        train_encoder=False,
        train_predictor=False,
        train_decoder=False,
    )
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


def prepare_dataloaders(batch_size: int, data_path: str, n_rollout=None, num_workers=4):
    transform = default_transform()
    datasets, _ = load_rearrange_slice_train_val(
        transform=transform,
        n_rollout=n_rollout,
        data_path=data_path,
        normalize_action=False,
        num_hist=3,
        num_pred=1,
        frameskip=1,
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        datasets["valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    num_actions = 6 #int(datasets["train"].get_all_actions().max().item() + 1)
    return train_loader, val_loader, num_actions


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = build_world_model(args.ckpt, args.img_size, args.num_hist, args.num_pred)
    wm.to(device)

    train_loader, val_loader, num_actions = prepare_dataloaders(
        args.batch_size, args.data_path, args.n_rollout, args.num_workers
    )

    classifier = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, num_actions),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        for obs, act, _ in train_loader:
            obs = {k: v.to(device) for k, v in obs.items()}
            act = act.to(device).squeeze(-1).long()  # (B,T)

            with torch.no_grad():
                z_obs = wm.encode_obs(obs)
                latent_act, _ = wm.encode_act(None, z_obs["visual_frame"])

            # Drop the first latent action (causal mask can't see next frame)
            # and pair the remaining T-1 vectors with ground-truth actions 0..T-2
            latent_act = latent_act.squeeze(2)[:, 1:, :]  # (B,T-1,10)
            labels = act[:, :-1]  # (B,T-1)

            logits = classifier(latent_act.reshape(-1, 10))
            loss = criterion(logits, labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} loss {loss.item():.4f}")

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for obs, act, _ in val_loader:
                obs = {k: v.to(device) for k, v in obs.items()}
                act = act.to(device).squeeze(-1).long()  # (B,T)
                z_obs = wm.encode_obs(obs)
                latent_act, _ = wm.encode_act(None, z_obs["visual_frame"])
                latent_act = latent_act.squeeze(2)[:, 1:, :]  # (B,T-1,10)
                labels = act[:, :-1]

                logits = classifier(latent_act.reshape(-1, 10))
                preds = logits.argmax(dim=-1)
                correct += (preds == labels.reshape(-1)).sum().item()
                total += preds.numel()
        acc = correct / total if total > 0 else 0.0
        print(f"Validation accuracy: {acc:.4f}")

        if epoch % 2 == 0:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "classifier_state": classifier.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "accuracy": acc,
                },
                save_dir / f"classifier_epoch_{epoch}.pth",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/content/drive/MyDrive/Model_checkpoints/2025-08-31/17-26-19/checkpoints/model_latest.pth")
    parser.add_argument("--data_path", type=str, default="/content/rearrange_2000")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_rollout", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_hist", type=int, default=3)
    parser.add_argument("--num_pred", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Model_checkpoints/classifier_ckpts")
    train_classifier(parser.parse_args())