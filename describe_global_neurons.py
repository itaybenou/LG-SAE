"""
Name global (image-level) SAE latents with CLIP-Dissect.

Each latent is described by the concept with the highest soft-WPMI score,
computed from CLIP image-text similarities of the images on which the latent
activates most.

--activations is an (N, U) tensor of global SAE codes for the probing images
ImageFolder(image_dir)[::sample_stride], in that order.

Pass --activations none to compute those codes from pre-extracted image
embeddings (--global_features, a single file or a shard directory in ImageFolder
order), save them to <output_dir>/z_global.pt, and then name the latents.

Example
  python describe_global_neurons.py --config configs/describe/global_clip.yaml \\
      --activations <z_global.pt> --image_dir <imagenet/val> --sample_stride 1 \\
      --concept_set concept_sets/10k.txt --output_dir <out_dir>

  python describe_global_neurons.py --config configs/describe/global_clip.yaml \\
      --activations none --sae_checkpoint <final.pt> --model_config <config.json> \\
      --global_features <val/global> --sae_batch_size 64 \\
      --image_dir <imagenet/val> --sample_stride 1 \\
      --concept_set concept_sets/10k.txt --output_dir <out_dir>
"""

import torch
from tqdm import tqdm

from lg_sae.config import Option, load_config
from lg_sae.dissect import soft_wpmi, top_activating_indices
from lg_sae.dissect.activations import resolve_activations
from lg_sae.dissect.common import (DESCRIBE_OPTIONS, cached, check_rows, encode_dataset, images_key,
                                   load_concepts, probe_dataset, save_descriptions, words_digest)
from lg_sae.third_party import clip

OPTIONS = DESCRIBE_OPTIONS + [
    Option("global_features", str, "Pre-extracted image embeddings, an (N, d) .pt/.pth file or a directory of "
           "such shards, in sorted ImageFolder order. Required when activations is null.", nullable=True),
    Option("clip_model", str, "OpenAI CLIP model.", choices=clip.available_models()),
]


@torch.no_grad()
def encode_text(model, words, batch_size, device):
    return torch.cat([model.encode_text(clip.tokenize(words[i:i + batch_size]).to(device)).float().cpu()
                      for i in tqdm(range(0, len(words), batch_size), desc="text features")])


def main():
    cfg = load_config(OPTIONS, description=__doc__)
    acts = resolve_activations(cfg, "global")
    words = load_concepts(cfg.concept_set)
    model, preprocess = clip.load(cfg.clip_model, device=cfg.device)
    model.eval()
    dataset = probe_dataset(cfg.image_dir, preprocess, cfg.sample_stride)
    check_rows(acts, dataset, cfg)
    print(f"activations {tuple(acts.shape)}, {len(words)} concepts, {len(dataset)} probing images")

    model_key = {"model": "openai_clip", "clip_model": cfg.clip_model}
    image_feats = cached(
        cfg.cache_dir, "clip_image_features", {**model_key, **images_key(cfg, len(dataset))},
        lambda: encode_dataset(model.encode_image, dataset, cfg.batch_size, cfg.num_workers, cfg.device))
    text_feats = cached(
        cfg.cache_dir, "clip_text_features", {**model_key, "words": words_digest(words)},
        lambda: encode_text(model, words, cfg.batch_size, cfg.device))
    del model
    torch.cuda.empty_cache()

    top_inds = top_activating_indices(acts, cfg.top_k, cfg.device, cfg.unit_batch_size)
    similarities = soft_wpmi(image_feats, text_feats, top_inds, softmax_temp=cfg.softmax_temp,
                             lam=cfg.lam, p_start=cfg.p_start, p_end=cfg.p_end, min_prob=cfg.min_prob,
                             device=cfg.device, unit_batch_size=cfg.unit_batch_size)
    save_descriptions(similarities, words, cfg)


if __name__ == "__main__":
    main()
