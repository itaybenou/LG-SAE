"""
Name local (patch-level) SAE latents with spatial CLIP-Dissect.

Each latent is described by the concept with the highest soft-WPMI score,
computed from CLIP-DINOiser dense image-text similarities at the image
locations where the latent activates most.

--activations is an (N, U, H, W) tensor of local SAE codes for the probing
images ImageFolder(image_dir)[::sample_stride], in that order, with (H, W)
equal to the CLIP-DINOiser feature grid (image_size / 16).

Pass --activations none to compute those codes from pre-extracted patch tokens
(--patch_features, a single file or a shard directory in ImageFolder order),
save them to <output_dir>/z_local.pt, and then name the latents.

Example
  python describe_local_neurons.py --config configs/describe/local_dinoiser.yaml \\
      --activations <z_local.pt> --image_dir <imagenet/val> --sample_stride 10 \\
      --concept_set concept_sets/20k.txt --dinoiser_ckpt <clip_dinoiser/last.pt> \\
      --output_dir <out_dir>

  python describe_local_neurons.py --config configs/describe/local_dinoiser.yaml \\
      --activations none --sae_checkpoint <final.pt> --model_config <config.json> \\
      --patch_features <val/patch_tokens> --sae_batch_size 64 \\
      --image_dir <imagenet/val> --sample_stride 10 \\
      --concept_set concept_sets/20k.txt --dinoiser_ckpt <clip_dinoiser/last.pt> \\
      --output_dir <out_dir>
"""

import os

import torch

from lg_sae.config import Option, load_config
from lg_sae.dissect import soft_wpmi, top_activating_indices
from lg_sae.dissect.activations import resolve_activations
from lg_sae.dissect.common import (DESCRIBE_OPTIONS, cached, check_rows, encode_dataset, images_key,
                                   load_concepts, probe_dataset, save_descriptions, words_digest)
from lg_sae.dissect.dinoiser import CLIPDINOiser, dense_preprocess

OPTIONS = DESCRIBE_OPTIONS + [
    Option("patch_features", str, "Pre-extracted patch tokens, an (N, T, d) .pt/.pth file or a directory of "
           "such shards, in sorted ImageFolder order. Required when activations is null.", nullable=True),
    Option("dinoiser_ckpt", str, "CLIP-DINOiser checkpoint (last.pt from the official release)."),
    Option("clip_model", str, "open_clip model name of the CLIP-DINOiser backbone."),
    Option("clip_pretrained", str, "open_clip pretrained tag of the backbone."),
    Option("gamma", float, "Threshold on CLIP-DINOiser patch correlations."),
    Option("image_size", int, "Resize + center-crop size of the probing images."),
]


def main():
    cfg = load_config(OPTIONS, description=__doc__)
    acts = resolve_activations(cfg, "local")
    words = load_concepts(cfg.concept_set)
    dataset = probe_dataset(cfg.image_dir, dense_preprocess(cfg.image_size), cfg.sample_stride)
    check_rows(acts, dataset, cfg)
    print(f"activations {tuple(acts.shape)}, {len(words)} concepts, {len(dataset)} probing images")

    model = CLIPDINOiser(cfg.dinoiser_ckpt, cfg.clip_model, cfg.clip_pretrained, cfg.gamma, cfg.device)
    grid = model.grid_size(cfg.image_size)
    if tuple(acts.shape[2:]) != grid:
        raise ValueError(f"Activation grid {tuple(acts.shape[2:])} != CLIP-DINOiser grid {grid} "
                         f"at image_size={cfg.image_size}")

    model_key = {"model": "clip_dinoiser", "clip_model": cfg.clip_model,
                 "clip_pretrained": cfg.clip_pretrained, "ckpt": os.path.abspath(cfg.dinoiser_ckpt)}
    image_feats = cached(
        cfg.cache_dir, "dinoiser_image_features",
        {**model_key, **images_key(cfg, len(dataset)), "gamma": cfg.gamma, "image_size": cfg.image_size},
        lambda: encode_dataset(model.encode_images, dataset, cfg.batch_size, cfg.num_workers, cfg.device))
    text_feats = cached(
        cfg.cache_dir, "dinoiser_text_features", {**model_key, "words": words_digest(words)},
        lambda: model.encode_text(words, cfg.batch_size))
    del model
    torch.cuda.empty_cache()

    top_inds = top_activating_indices(acts, cfg.top_k, cfg.device, cfg.unit_batch_size)
    del acts
    location_feats = image_feats.permute(0, 2, 3, 1).reshape(-1, image_feats.shape[1])
    similarities = soft_wpmi(location_feats, text_feats, top_inds, softmax_temp=cfg.softmax_temp,
                             lam=cfg.lam, p_start=cfg.p_start, p_end=cfg.p_end, min_prob=cfg.min_prob,
                             device=cfg.device, unit_batch_size=cfg.unit_batch_size)
    save_descriptions(similarities, words, cfg)


if __name__ == "__main__":
    main()
