"""
Train an LG-SAE (local SAE + global SAE + bridge) on pre-extracted features.

Stages
  1. Local SAE on patch tokens                 (MSE + AuxK)
  2. Global SAE on image-level embeddings      (MSE + AuxK)
  3. Bridge C, b with both SAEs frozen         (L_link + L_rec_bridge + L1(C))
  4. ft_rounds x [4a local-SAE fine-tuning, 4b bridge fine-tuning]

Stages 1-3 are skipped when the matching --local_ckpt / --global_ckpt /
--bridge_ckpt is given (checkpoints from this repo or from overcomplete).

Features come from save_shards.py: --global_dir holds (N, d_global) tensors
(CLIP image embeddings or DINOv2 CLS tokens), --patch_dir holds the matching
(N, T, d_local) patch tokens. Dimensions are read from the data.

Example
  python train_lg_sae.py --config configs/train/clip_batchtopk.yaml \\
      --global_dir /data/clip/global --patch_dir /data/clip/patch_tokens \\
      --save_dir runs/clip_batchtopk --device cuda:0
"""

import json
import os
import random

import torch

from lg_sae.config import Option, config_to_json, load_config
from lg_sae.data import ShardDataset, inspect_shards
from lg_sae.sae import (ENCODER_TYPES, POOLINGS, SAE_TYPES, LocalGlobalSAE,
                        build_sae, load_components)
from lg_sae.trainer import finetune_local, train_bridge, train_sae_component

OPTIONS = [
    # Data / run
    Option("backbone", str, "Backbone that produced the features.", choices=("clip", "dinov2")),
    Option("global_dir", str, "Global features: a .pt file or a directory of shards."),
    Option("patch_dir", str, "Patch tokens: a .pt file or a directory of shards."),
    Option("save_dir", str, "Output directory for checkpoints, logs and the resolved config."),
    Option("device", str, "Torch device, e.g. cuda:0."),
    Option("seed", int, "Random seed (init and data shuffling)."),
    Option("clip_grad", float, "Max global grad norm (0 disables clipping)."),
    # Model
    Option("sae_type", str, "Sparsity mechanism of both SAEs.", choices=SAE_TYPES),
    Option("n_local", int, "Number of local (patch) concepts."),
    Option("n_global", int, "Number of global (image) concepts."),
    Option("k_local", int, "Active local concepts per patch token (mean over the batch for batchtopk)."),
    Option("k_global", int, "Active global concepts per image (mean over the batch for batchtopk)."),
    Option("local_encoder", str, "Local SAE encoder.", choices=ENCODER_TYPES),
    Option("global_encoder", str, "Global SAE encoder.", choices=ENCODER_TYPES),
    Option("threshold_momentum", float, "EMA momentum of the BatchTopK eval threshold (batchtopk only).",
           nullable=True),
    Option("pooling", str, "Pooling of patch codes before the bridge.", choices=POOLINGS),
    Option("c_top_k", int, "Keep only the top-k entries per row of C (null = dense).", nullable=True),
    # Stage 1: local SAE
    Option("local_epochs", int, "Stage 1 epochs."),
    Option("local_lr", float, "Stage 1 learning rate."),
    Option("local_batch_size", int, "Stage 1 batch size (patch tokens)."),
    Option("local_aux_penalty", float, "Stage 1 AuxK weight."),
    Option("local_log_interval", int, "Stage 1 steps between prints (0 = epoch end only)."),
    Option("local_ckpt", str, "Load the local SAE from this checkpoint and skip stage 1.", nullable=True),
    # Stage 2: global SAE
    Option("global_epochs", int, "Stage 2 epochs."),
    Option("global_lr", float, "Stage 2 learning rate."),
    Option("global_batch_size", int, "Stage 2 batch size (images)."),
    Option("global_aux_penalty", float, "Stage 2 AuxK weight."),
    Option("global_log_interval", int, "Stage 2 steps between prints (0 = epoch end only)."),
    Option("global_ckpt", str, "Load the global SAE from this checkpoint and skip stage 2.", nullable=True),
    # Stage 3 (and 4b): bridge
    Option("bridge_epochs", int, "Stage 3 epochs."),
    Option("bridge_lr", float, "Stage 3 learning rate."),
    Option("bridge_batch_size", int, "Stage 3 / 4b batch size (images)."),
    Option("bridge_w_rec", float, "Weight of L_rec_bridge in stages 3 / 4b."),
    Option("bridge_l1_C", float, "Weight of the L1 penalty on C in stages 3 / 4b."),
    Option("bridge_log_interval", int, "Stage 3 / 4b steps between prints (0 = epoch end only)."),
    Option("bridge_ckpt", str, "Load the bridge from this checkpoint and skip stage 3.", nullable=True),
    # Stage 4: iterative fine-tuning
    Option("ft_rounds", int, "Number of (4a, 4b) rounds; 0 disables stage 4."),
    Option("ft_local_epochs", int, "Stage 4a epochs per round."),
    Option("ft_local_lr", float, "Stage 4a learning rate."),
    Option("ft_local_batch_size", int, "Stage 4a batch size (images)."),
    Option("ft_local_aux_penalty", float, "Stage 4a AuxK weight."),
    Option("ft_w_rec_bridge", float, "Stage 4a weight of L_rec_bridge."),
    Option("ft_w_link", float, "Stage 4a weight of L_link."),
    Option("ft_freeze_local_dictionary", bool, "Stage 4a trains only the local encoder."),
    Option("ft_bridge_epochs", int, "Stage 4b epochs per round."),
    Option("ft_bridge_lr", float, "Stage 4b learning rate."),
    Option("ft_log_interval", int, "Stage 4a steps between prints (0 = epoch end only)."),
]


def banner(title):
    print("=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


def save_checkpoint(model, cfg, save_dir, filename):
    path = os.path.join(save_dir, filename)
    torch.save({"model": model.state_dict(), "config": vars(cfg)}, path)
    print(f"  saved {path}")


def save_logs(logs, save_dir, filename):
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(logs, f, indent=2)


def main():
    cfg = load_config(OPTIONS, description=__doc__)
    if cfg.sae_type == "batchtopk" and cfg.threshold_momentum is None:
        raise ValueError("threshold_momentum must be set for sae_type=batchtopk")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        f.write(config_to_json(cfg))

    info = inspect_shards(cfg.global_dir, cfg.patch_dir)
    banner(f"LG-SAE | backbone={cfg.backbone} sae_type={cfg.sae_type}")
    print(f"  data: {info.n_samples} images in {info.n_shards} shard(s), {info.n_patches} patches/image, "
          f"d_global={info.global_dim}, d_local={info.local_dim}")
    print(f"  model: n_local={cfg.n_local} (k={cfg.k_local}, {cfg.local_encoder}), "
          f"n_global={cfg.n_global} (k={cfg.k_global}, {cfg.global_encoder}), "
          f"pooling={cfg.pooling}, c_top_k={cfg.c_top_k}")

    device = cfg.device
    model = LocalGlobalSAE(
        local_sae=build_sae(cfg.sae_type, info.local_dim, cfg.n_local, cfg.k_local,
                            cfg.local_encoder, cfg.threshold_momentum, device),
        global_sae=build_sae(cfg.sae_type, info.global_dim, cfg.n_global, cfg.k_global,
                             cfg.global_encoder, cfg.threshold_momentum, device),
        pooling=cfg.pooling,
        c_top_k=cfg.c_top_k,
    )

    def dataset(mode, batch_size):
        return ShardDataset(cfg.global_dir, cfg.patch_dir, mode, batch_size, shuffle=True)

    # Stage 1: local SAE
    if cfg.local_ckpt is not None:
        banner("STAGE 1: local SAE (loaded)")
        n = load_components(model, cfg.local_ckpt, ["local"], device)
        print(f"  loaded {n} tensors from {cfg.local_ckpt}")
    else:
        banner(f"STAGE 1: local SAE | epochs={cfg.local_epochs} lr={cfg.local_lr} "
               f"batch={cfg.local_batch_size} tokens aux={cfg.local_aux_penalty}")
        logs = train_sae_component(
            model, "local", dataset("patch", cfg.local_batch_size), epochs=cfg.local_epochs,
            lr=cfg.local_lr, aux_penalty=cfg.local_aux_penalty, clip_grad=cfg.clip_grad,
            device=device, log_interval=cfg.local_log_interval, name="stage1-local")
        save_logs(logs, cfg.save_dir, "logs_stage1.json")
        save_checkpoint(model, cfg, cfg.save_dir, "checkpoint_stage1.pt")

    # Stage 2: global SAE
    if cfg.global_ckpt is not None:
        banner("STAGE 2: global SAE (loaded)")
        n = load_components(model, cfg.global_ckpt, ["global"], device)
        print(f"  loaded {n} tensors from {cfg.global_ckpt}")
    else:
        banner(f"STAGE 2: global SAE | epochs={cfg.global_epochs} lr={cfg.global_lr} "
               f"batch={cfg.global_batch_size} aux={cfg.global_aux_penalty}")
        logs = train_sae_component(
            model, "global", dataset("global", cfg.global_batch_size), epochs=cfg.global_epochs,
            lr=cfg.global_lr, aux_penalty=cfg.global_aux_penalty, clip_grad=cfg.clip_grad,
            device=device, log_interval=cfg.global_log_interval, name="stage2-global")
        save_logs(logs, cfg.save_dir, "logs_stage2.json")
        save_checkpoint(model, cfg, cfg.save_dir, "checkpoint_stage2.pt")

    # Stage 3: bridge
    bridge_kwargs = dict(w_rec_bridge=cfg.bridge_w_rec, l1_penalty=cfg.bridge_l1_C,
                         clip_grad=cfg.clip_grad, device=device, log_interval=cfg.bridge_log_interval)
    if cfg.bridge_ckpt is not None:
        banner("STAGE 3: bridge (loaded)")
        n = load_components(model, cfg.bridge_ckpt, ["bridge"], device)
        print(f"  loaded {n} tensors from {cfg.bridge_ckpt}")
    else:
        banner(f"STAGE 3: bridge | epochs={cfg.bridge_epochs} lr={cfg.bridge_lr} "
               f"batch={cfg.bridge_batch_size} w_rec={cfg.bridge_w_rec} l1_C={cfg.bridge_l1_C}")
        logs = train_bridge(model, dataset("paired", cfg.bridge_batch_size), epochs=cfg.bridge_epochs,
                            lr=cfg.bridge_lr, name="stage3-bridge", **bridge_kwargs)
        save_logs(logs, cfg.save_dir, "logs_stage3.json")
        save_checkpoint(model, cfg, cfg.save_dir, "checkpoint_stage3.pt")

    # Stage 4: iterative fine-tuning
    if cfg.ft_rounds > 0:
        banner(f"STAGE 4: {cfg.ft_rounds} rounds of local FT ({cfg.ft_local_epochs} ep, "
               f"lr={cfg.ft_local_lr}, w_rec_bridge={cfg.ft_w_rec_bridge}, w_link={cfg.ft_w_link}, "
               f"freeze_D_local={cfg.ft_freeze_local_dictionary}) + bridge FT "
               f"({cfg.ft_bridge_epochs} ep, lr={cfg.ft_bridge_lr})")
        logs = {"local_ft": [], "bridge_ft": []}
        for r in range(1, cfg.ft_rounds + 1):
            print(f"\n  --- round {r}/{cfg.ft_rounds} ---")
            logs["local_ft"].append(finetune_local(
                model, dataset("paired", cfg.ft_local_batch_size), epochs=cfg.ft_local_epochs,
                lr=cfg.ft_local_lr, aux_penalty=cfg.ft_local_aux_penalty,
                w_rec_bridge=cfg.ft_w_rec_bridge, w_link=cfg.ft_w_link,
                freeze_local_dictionary=cfg.ft_freeze_local_dictionary, clip_grad=cfg.clip_grad,
                device=device, log_interval=cfg.ft_log_interval, name=f"stage4a-local-r{r}"))
            logs["bridge_ft"].append(train_bridge(
                model, dataset("paired", cfg.bridge_batch_size), epochs=cfg.ft_bridge_epochs,
                lr=cfg.ft_bridge_lr, name=f"stage4b-bridge-r{r}", **bridge_kwargs))
            save_logs(logs, cfg.save_dir, "logs_stage4.json")
            save_checkpoint(model, cfg, cfg.save_dir, f"checkpoint_stage4_round{r}.pt")

    banner("DONE")
    save_checkpoint(model, cfg, cfg.save_dir, "final.pt")


if __name__ == "__main__":
    main()
