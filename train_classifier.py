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

    # --- resume from checkpoint if available ---
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        classifier.load_state_dict(ckpt["classifier_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}, val acc={ckpt.get('accuracy', 0):.4f}")

    # --- training loop ---
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        classifier.train()
        running_loss = 0.0
        for step, (obs, act, _) in enumerate(train_loader, start=1):
            obs = {k: v.to(device) for k, v in obs.items()}
            act = act.to(device).squeeze(-1).long()  # (B,T)

            with torch.no_grad():
                z_obs = wm.encode_obs(obs)
                latent_act, _ = wm.encode_act(None, z_obs["visual_frame"])

            latent_act = latent_act.squeeze(2)[:, 1:, :]  # (B,T-1,10)
            labels = act[:, :-1]  # (B,T-1)

            logits = classifier(latent_act.reshape(-1, 10))
            loss = criterion(logits, labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            running_loss += loss.item()
            global_step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                print(f"[Epoch {epoch:03d} | Step {step:04d}] Loss {avg_loss:.4f}")
                running_loss = 0.0

        # --- validation ---
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for obs, act, _ in val_loader:
                obs = {k: v.to(device) for k, v in obs.items()}
                act = act.to(device).squeeze(-1).long()
                z_obs = wm.encode_obs(obs)
                latent_act, _ = wm.encode_act(None, z_obs["visual_frame"])
                latent_act = latent_act.squeeze(2)[:, 1:, :]
                labels = act[:, :-1]

                logits = classifier(latent_act.reshape(-1, 10))
                preds = logits.argmax(dim=-1)
                correct += (preds == labels.reshape(-1)).sum().item()
                total += preds.numel()

        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch} finished. Val acc={acc:.4f}")

        # --- save checkpoint ---
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"classifier_epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "classifier_state": classifier.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "accuracy": acc,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_rollout", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_hist", type=int, default=3)
    parser.add_argument("--num_pred", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./classifier_ckpts")
    parser.add_argument("--resume", type=str, default=None, help="Path to classifier checkpoint to resume from")
    parser.add_argument("--log_every", type=int, default=50, help="Steps per log print")
    train_classifier(parser.parse_args())
