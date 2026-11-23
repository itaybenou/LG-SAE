"""
PartImageNet composition evaluation of an LG-SAE bridge.

Matches every PartImageNet supercategory to a global concept (i*, point-biserial)
and every part to a local concept (canonical j* with a token-frequency cap), then
scores the edges predicted by the bridge, ``C[i*(g), j*(p)] > tau``, against the
ground-truth ontology. See ``lg_sae/composition.py`` for the exact definitions.

Works for any backbone (CLIP, DINOv2): the patch grid is sqrt(T) and the input
dimensions are read from the features. Works for TopK and BatchTopK SAEs: the
model is built from ``--model_config`` (e.g. the ``config.json`` of a training
run) and evaluated in eval mode (BatchTopK uses its running threshold).

Inputs
  --patch_tokens / --global_features / --image_ids : PartImageNet test features
      in the same row order (see extract_partimagenet_features.py)
  --labelmap_dir : PartImageNet test label PNGs (<image_id>.png)
  --ontology     : gt_ontology.json (COCO-style categories with supercategories)
  --freq_patch_file : patch tokens of the SAE's training data (e.g. one training
      shard); its first --freq_n_images images define the token frequency

Output: <output_dir>/results.json with precision, recall, F1 and TP/FP/FN.

Example
  python eval_partimagenet_composition.py --config configs/eval/partimagenet_composition.yaml \\
      --checkpoint saved_models/clip_batchtopk/final.pt \\
      --model_config saved_models/clip_batchtopk/config.json \\
      --patch_tokens /data/partimagenet/clip_vitb16/patch_tokens.pt \\
      --global_features /data/partimagenet/clip_vitb16/global.pt \\
      --image_ids /data/partimagenet/clip_vitb16/image_ids.json \\
      --labelmap_dir /data/PartImageNet/annotations/test \\
      --ontology /data/PartImageNet/gt_ontology.json \\
      --freq_patch_file /data/clip_vitb16/train/patch_tokens/shard_00000.pt \\
      --output_dir PartImageNet_composition_eval/clip_batchtopk
"""

import json
import os

import torch
import yaml

from lg_sae.composition import (edge_metrics, encode_global, local_token_frequency,
                                part_cosine, select_i_star, select_j_star)
from lg_sae.config import Option, load_config
from lg_sae.partimagenet import (image_supercategories, load_features, load_grid_labels,
                                 load_ontology, patch_grid_size)
from lg_sae.sae import COMPONENTS, LocalGlobalSAE, build_sae, load_components

OPTIONS = [
    # Model
    Option("checkpoint", str, "LG-SAE checkpoint (this repo or overcomplete)."),
    Option("model_config", str, "JSON/YAML with the model hyperparameters (the config.json of a "
                                "training run works as is)."),
    # PartImageNet test set
    Option("patch_tokens", str, "PartImageNet patch tokens, (N, T, d_local) .pt file."),
    Option("global_features", str, "PartImageNet global features, (N, d_global) .pt file."),
    Option("image_ids", str, "JSON list of the N image ids, in feature row order."),
    Option("labelmap_dir", str, "Directory of PartImageNet label maps (<image_id>.png)."),
    Option("ontology", str, "PartImageNet gt_ontology.json."),
    Option("background_label", int, "Label-map value of background pixels."),
    # Frequency cap
    Option("freq_patch_file", str, "Patch tokens (N, T, d_local) from the SAE's training data "
                                   "used to measure local token frequencies."),
    Option("freq_n_images", int, "Number of leading images of freq_patch_file to use."),
    Option("freq_cap", float, "Local concepts active on more than this fraction of tokens are "
                              "excluded from j* (1.0 disables the cap)."),
    # j* selection and scoring
    Option("token_top_m", int, "Codes kept per token when computing j*."),
    Option("min_active_patches", int, "Minimum active PartImageNet tokens of a j* candidate."),
    Option("tau", float, "An edge is predicted when C[i*, j*] > tau."),
    Option("eps", float, "Numerical epsilon of the cosine and point-biserial denominators."),
    # Run
    Option("batch_size", int, "Images per encoding batch."),
    Option("device", str, "Torch device, e.g. cuda:0."),
    Option("output_dir", str, "Directory that receives results.json."),
]

MODEL_KEYS = ("sae_type", "n_local", "n_global", "k_local", "k_global", "local_encoder",
              "global_encoder", "threshold_momentum", "pooling", "c_top_k")


def load_model(checkpoint, model_config, local_dim, global_dim, device):
    with open(model_config) as f:
        mc = yaml.safe_load(f)
    missing = [k for k in MODEL_KEYS if k not in mc]
    if missing:
        raise KeyError(f"{model_config} is missing model keys {missing}")
    model = LocalGlobalSAE(
        local_sae=build_sae(mc["sae_type"], local_dim, mc["n_local"], mc["k_local"],
                            mc["local_encoder"], mc["threshold_momentum"], device),
        global_sae=build_sae(mc["sae_type"], global_dim, mc["n_global"], mc["k_global"],
                             mc["global_encoder"], mc["threshold_momentum"], device),
        pooling=mc["pooling"], c_top_k=mc["c_top_k"])
    load_components(model, checkpoint, COMPONENTS, device)
    return model.eval()


@torch.no_grad()
def main():
    cfg = load_config(OPTIONS, description=__doc__)

    ontology = load_ontology(cfg.ontology)
    patch_tokens, global_features, image_ids = load_features(
        cfg.patch_tokens, cfg.global_features, cfg.image_ids)
    grid = patch_grid_size(patch_tokens.shape[1])
    labels = load_grid_labels(image_ids, cfg.labelmap_dir, grid, ontology.n_parts, cfg.background_label)
    supercats = image_supercategories(labels, ontology.part_to_supercat)
    print(f"{len(image_ids)} images, {grid}x{grid} patch grid, {ontology.n_parts} parts in "
          f"{ontology.n_supercats} supercategories ({int((supercats < 0).sum())} images without parts)")

    model = load_model(cfg.checkpoint, cfg.model_config, patch_tokens.shape[2],
                       global_features.shape[1], cfg.device)
    print(f"{cfg.checkpoint}: {type(model.local_sae).__name__}, "
          f"n_local={model.n_local}, n_global={model.n_global}")

    freq_tokens = torch.load(cfg.freq_patch_file, map_location="cpu", weights_only=True, mmap=True)
    if freq_tokens.shape[0] < cfg.freq_n_images:
        raise ValueError(f"{cfg.freq_patch_file} has {freq_tokens.shape[0]} images "
                         f"< freq_n_images={cfg.freq_n_images}")
    token_freq = local_token_frequency(model, freq_tokens[:cfg.freq_n_images], cfg.batch_size, cfg.device)
    excluded = token_freq > cfg.freq_cap
    print(f"frequency cap {cfg.freq_cap}: {int(excluded.sum())} local concepts excluded from j*")

    z_global = encode_global(model, global_features, cfg.batch_size, cfg.device)
    i_star, supercat_valid = select_i_star(z_global, supercats, ontology.n_supercats, cfg.eps)
    mean_cos, active_tokens = part_cosine(model, patch_tokens, labels, ontology.n_parts, excluded,
                                          cfg.token_top_m, cfg.eps, cfg.batch_size, cfg.device)
    j_star, part_valid = select_j_star(mean_cos, active_tokens, cfg.min_active_patches)

    metrics = edge_metrics(model.get_C().float().cpu().numpy(), i_star, supercat_valid,
                           j_star, part_valid, ontology.ground_truth(), cfg.tau)
    if metrics["n_skipped_pairs"]:
        print(f"warning: {metrics['n_skipped_pairs']} (supercategory, part) pairs skipped "
              f"(supercategory or part absent from the data)")

    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, "results.json")
    with open(path, "w") as f:
        json.dump({**metrics, "config": vars(cfg)}, f, indent=2)
    print(f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
          f"F1={metrics['f1']:.4f} (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
