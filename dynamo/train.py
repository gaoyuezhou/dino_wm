import os
import tqdm
import utils
import hydra
import torch
import einops
import datasets
import numpy as np
import torch.distributed
from pathlib import Path
from datetime import timedelta
from omegaconf import OmegaConf
from accelerate import Accelerator
from collections import OrderedDict
from workspaces.base import Workspace
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs

os.environ["WANDB_START_METHOD"] = "thread"
logger = get_logger(__name__)


class Trainer:
    def __init__(self, cfg):
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=cfg.timeout_seconds)
        )
        dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.cfg = cfg
        self.effective_batch_size = self.cfg.batch_size
        self.accelerator = Accelerator(
            log_with="wandb", kwargs_handlers=[process_group_kwargs, dist_kwargs]
        )
        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
        utils.set_seed_everywhere(cfg.seed)

        self.job_num, self.work_dir = utils.get_hydra_jobnum_workdir()

        # all processes use the work_dir from the main process
        if torch.distributed.is_initialized():
            objs = [str(self.work_dir)]
            torch.distributed.broadcast_object_list(objs, 0)
            self.work_dir = Path(objs[0])
        self.accelerator.wait_for_everyone()
        logger.info("Saving to {}".format(self.work_dir))
        os.chdir(self.work_dir)
        self.work_dir = Path(os.getcwd())  # get the absolute path

        self.dataset = hydra.utils.instantiate(cfg.env.dataset)
        self.train_set, self.test_set = self._split_and_slice_dataset(self.dataset)
        self._setup_loaders(batch_size=self.cfg.batch_size)
        self._init_tracker(cfg)

        # Create the model
        self.encoder = None
        self.projector = None
        self.ssl = None
        self._init_encoder()
        self._init_projector()
        self._init_ssl()

        self.workspace: Workspace = hydra.utils.instantiate(
            self.cfg.env.workspace,
            cfg=self.cfg,
            work_dir=self.work_dir,
            _recursive_=False,
        )
        self.workspace.set_dataset(self.dataset)

        self.log_components = OrderedDict()
        self.epoch = 0

    def _init_tracker(self, cfg):
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_cfg["effective_batch_size"] = self.effective_batch_size
        wandb_cfg["save_path"] = str(self.work_dir)
        self.accelerator.init_trackers(
            project_name=cfg.project,
            config=wandb_cfg,
            init_kwargs={
                "wandb": {
                    "reinit": False,
                    "settings": {"start_method": "thread"},
                },
            },
        )
        if self.accelerator.is_main_process:
            self.wandb_run = self.accelerator.get_tracker("wandb", unwrap=True)
            logger.info("wandb run url: %s", self.wandb_run.get_url())

    def _init_encoder(self):
        if self.encoder is None:  # possibly already initialized from snapshot
            self.encoder = hydra.utils.instantiate(self.cfg.encoder)
            if self.cfg.sync_bn:
                self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.encoder
                )
            self.encoder_optim = torch.optim.AdamW(
                params=self.encoder.parameters(),
                lr=self.cfg.ssl_lr,
                weight_decay=self.cfg.ssl_weight_decay,
                betas=tuple(self.cfg.betas),
            )
            (
                self.encoder,
                self.encoder_optim,
            ) = self.accelerator.prepare(self.encoder, self.encoder_optim)
            if self.accelerator.is_main_process:
                self.wandb_run.watch(self.encoder)

    def _init_projector(self):
        if self.projector is None:  # possibly already initialized from snapshot
            self.projector = hydra.utils.instantiate(
                self.cfg.projector, _recursive_=False
            )
            self.projector_optim: torch.optim.Optimizer = (
                self.projector.configure_optimizers(
                    lr=self.cfg.ssl_lr,
                    weight_decay=self.cfg.ssl_weight_decay,
                    betas=tuple(self.cfg.betas),
                )
            )
            (
                self.projector,
                self.projector_optim,
            ) = self.accelerator.prepare(self.projector, self.projector_optim)

    def _init_ssl(self):
        if self.ssl is None:
            self.ssl = hydra.utils.instantiate(
                self.cfg.ssl,
                encoder=self.encoder,
                projector=self.projector,
            )

    def _split_and_slice_dataset(self, dataset):
        kwargs = {
            "train_fraction": self.cfg.train_fraction,
            "random_seed": self.cfg.seed,
            "window_size": self.cfg.window_size,
            "future_conditional": (self.cfg.goal_conditional == "future"),
            "min_future_sep": self.cfg.min_future_sep,
            "future_seq_len": self.cfg.goal_seq_len,
            "num_extra_predicted_actions": self.cfg.num_extra_predicted_actions,
        }
        return datasets.core.get_train_val_sliced(dataset, **kwargs)

    def _setup_loaders(self, batch_size=None, pin_memory=True, num_workers=None):
        if num_workers is None:
            num_workers = self.cfg.num_workers
        kwargs = {
            "batch_size": batch_size or self.cfg.batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        # scale batch size by number of gpus
        assert kwargs["batch_size"] % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Got {kwargs['batch_size']} and {self.accelerator.num_processes}."
        )
        kwargs["batch_size"] = kwargs["batch_size"] // self.accelerator.num_processes
        self.train_loader = DataLoader(self.train_set, shuffle=True, **kwargs)
        self.test_loader = DataLoader(self.test_set, shuffle=False, **kwargs)

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare(self.test_loader)

    def train(self):
        if self.cfg.use_lr_scheduling:
            lr = self.adjust_lr()
            self.log_append("metrics", 1, {"lr": lr})
        self.ssl.adjust_beta(self.epoch, self.cfg.num_epochs)
        pbar = tqdm.tqdm(
            self.train_loader,
            desc=f"Training epoch {self.epoch}",
            disable=not self.accelerator.is_main_process,
            ncols=80,
        )
        for data in pbar:
            obs, _, _ = data

            with self.accelerator.autocast():
                (
                    obs_enc,
                    obs_proj,
                    ssl_loss,
                    ssl_loss_components,
                ) = self.ssl.forward(obs)
            self.log_append("ssl_train", len(obs), ssl_loss_components)
            self.accelerator.backward(ssl_loss, retain_graph=True)

            if self.cfg.clip_grad_norm:
                self.accelerator.clip_grad_norm_(
                    self.encoder.parameters(), self.cfg.clip_grad_norm
                )
                self.accelerator.clip_grad_norm_(
                    self.projector.parameters(), self.cfg.clip_grad_norm
                )
                self.accelerator.clip_grad_norm_(
                    self.ssl.parameters(), self.cfg.clip_grad_norm
                )

            self.encoder_optim.step()
            self.projector_optim.step()
            self.ssl.step()

            self.encoder_optim.zero_grad(set_to_none=True)
            self.projector_optim.zero_grad(set_to_none=True)

    def eval(self):
        if self.cfg.eval_offline:
            # env-specific offline eval
            self.workspace.set_models(
                encoder=self.encoder,
                projector=self.projector,
            )
            offline_eval_results = self.workspace.run_offline_eval()
            if self.accelerator.is_main_process:
                self.log_append("env_offline_eval", 1, offline_eval_results)

        with utils.inference.eval_mode(
            self.encoder,
            self.projector,
            no_grad=True,
        ):
            # eval on test set
            self.eval_loss = 0
            for data in self.test_loader:
                obs, _, _ = data

                (
                    obs_enc,
                    obs_proj,
                    ssl_loss,
                    ssl_loss_components,
                ) = self.ssl.forward(obs)
                ssl_loss = self.accelerator.gather_for_metrics(ssl_loss).mean()
                ssl_loss_components = utils.reduce_dict(
                    torch.mean,
                    self.accelerator.gather_for_metrics(ssl_loss_components),
                )
                self.log_append(
                    "ssl_eval",
                    len(obs),
                    ssl_loss_components,
                )

                flat_obs_enc = self.accelerator.gather_for_metrics(obs_enc)
                flat_obs_enc = einops.rearrange(flat_obs_enc, "N T V E -> (N T V) E")
                obs_enc_mean_std = flat_obs_enc.std(dim=0).mean()
                obs_enc_mean_norm = flat_obs_enc.norm(dim=-1).mean()
                self.log_append(
                    "metrics",
                    len(flat_obs_enc),
                    {
                        "obs_enc_mean_std": obs_enc_mean_std,
                        "obs_enc_mean_norm": obs_enc_mean_norm,
                    },
                )

                flat_obs_proj = self.accelerator.gather_for_metrics(obs_proj)
                flat_obs_proj = einops.rearrange(flat_obs_proj, "N T V Z -> (N T V) Z")
                obs_proj_mean_std = flat_obs_proj.std(dim=0).mean()
                obs_proj_mean_norm = flat_obs_proj.norm(dim=-1).mean()
                self.log_append(
                    "metrics",
                    len(flat_obs_proj),
                    {
                        "obs_proj_mean_std": obs_proj_mean_std,
                        "obs_proj_mean_norm": obs_proj_mean_norm,
                    },
                )

    def run(self):
        snapshot = Path(self.work_dir) / "snapshot.pt"
        if snapshot.exists():
            print(f"Resuming: {snapshot}")
            self.load_snapshot()

        self.train_iterator = tqdm.trange(
            self.epoch,
            self.cfg.num_epochs,
            disable=not self.accelerator.is_main_process,
            ncols=80,
        )
        self.train_iterator.set_description("Training")
        # Reset the log.
        self.log_components = OrderedDict()
        for epoch in self.train_iterator:
            self.epoch = epoch
            self.train()
            self.eval()
            self.flush_log(step=self.epoch, iterator=self.train_iterator)
            if (self.epoch + 1) % self.cfg.save_every_epochs == 0:
                self.save_snapshot()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

        return float(self.eval_loss)

    def save_snapshot(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._keys_to_save = [
                "encoder",
                "projector",
                "encoder_optim",
                "projector_optim",
                "ssl",
                "epoch",
            ]
            payload = {}
            # if key is an accelerator DDP model, unwrap
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    payload[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    payload[k] = self.__dict__[k]
            with (self.work_dir / "snapshot.pt").open("wb") as f:
                torch.save(payload, f)
            with (self.work_dir / "encoder.pt").open("wb") as f:
                torch.save(payload["encoder"], f)
            with (self.work_dir / f"snapshot_{self.epoch}.pt").open("wb") as f:
                torch.save(payload, f)
            with (self.work_dir / f"encoder_{self.epoch}.pt").open("wb") as f:
                torch.save(payload["encoder"], f)

    def load_snapshot(self):
        with (self.work_dir / "snapshot.pt").open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        not_in_payload = set(self._keys_to_save) - set(payload.keys())
        if len(not_in_payload):
            logger.warning("Keys not found in snapshot: %s", not_in_payload)

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value),
            )

    def flush_log(self, step, iterator=None):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        if iterator is not None:
            iterator.set_postfix_str(postfix)
        self.accelerator.log(log_components, step=step)
        logger.info(f"[{self.job_num}] Epoch {self.epoch}: {log_components}")
        self.log_components = OrderedDict()

    def adjust_lr(self):
        # from https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L420
        """Decays the learning rate with half-cycle cosine after warmup"""
        # fmt: off
        if self.epoch < self.cfg.warmup_epochs:
            lr = self.cfg.ssl_lr * self.epoch / self.cfg.warmup_epochs
        else:
            lr = self.cfg.ssl_lr * 0.5 * (1.0 + np.cos(np.pi * (self.epoch - self.cfg.warmup_epochs) / (self.cfg.num_epochs - self.cfg.warmup_epochs)))
        # fmt: on
        optimizers = [self.encoder_optim, self.projector_optim]
        for optim in optimizers:
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        return lr


@hydra.main(version_base="1.2", config_path="configs", config_name="train")
def main(cfg):
    trainer = Trainer(cfg)
    eval_loss = trainer.run()
    return eval_loss


if __name__ == "__main__":
    main()
