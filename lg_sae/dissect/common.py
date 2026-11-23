"""Shared pieces of the describe_{local,global}_neurons scripts."""

import hashlib
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from lg_sae.config import Option, config_to_json

DESCRIBE_OPTIONS = [
    Option("activations", str, "Saved SAE activations .pt whose rows are ImageFolder(image_dir)[::sample_stride]. "
           "null computes them from the pre-extracted features and saves them in output_dir.", nullable=True),
    Option("sae_checkpoint", str, "LG-SAE checkpoint (this repo or overcomplete). Required when activations is null.",
           nullable=True),
    Option("model_config", str, "JSON or YAML with the SAE hyperparameters (a training config.json or "
           "configs/train/*.yaml). Required when activations is null.", nullable=True),
    Option("sae_batch_size", int, "Images per SAE forward pass when computing activations. "
           "Required when activations is null.", nullable=True),
    Option("image_dir", str, "ImageFolder root of the probing images the activations were computed on."),
    Option("sample_stride", int, "Every sample_stride-th image of image_dir (sorted ImageFolder order)."),
    Option("concept_set", str, "Text file with one concept per line."),
    Option("layer_name", str, "Value of the 'layer' column in descriptions.csv."),
    Option("output_dir", str, "Directory for descriptions.csv and config.json."),
    Option("cache_dir", str, "Directory to cache CLIP image/text features (null = no caching).", nullable=True),
    Option("batch_size", int, "Images (or text prompts) per CLIP forward pass."),
    Option("num_workers", int, "DataLoader workers for the probing images."),
    Option("device", str, "Torch device, e.g. cuda:0."),
    Option("top_k", int, "Top-activating probing rows (images / locations) used per unit."),
    Option("softmax_temp", float, "Scale of the image-text cosine similarities before the concept softmax."),
    Option("lam", float, "Weight of the concept prior term in soft-WPMI."),
    Option("p_start", float, "Example probability of a unit's top row."),
    Option("p_end", float, "Example probability of a unit's top_k-th row."),
    Option("min_prob", float, "Additive constant inside the soft-WPMI log."),
    Option("unit_batch_size", int, "Units processed together on the device."),
]


def load_concepts(path):
    with open(path) as f:
        return [word for word in f.read().split("\n") if word != ""]


def load_activations(path, ndim):
    acts = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(acts) or acts.dim() != ndim:
        shape = tuple(acts.shape) if torch.is_tensor(acts) else type(acts).__name__
        raise ValueError(f"Expected a {ndim}-D activation tensor in {path}, got {shape}")
    return acts


def probe_dataset(image_dir, transform, sample_stride):
    dataset = datasets.ImageFolder(image_dir, transform)
    if sample_stride > 1:
        dataset = Subset(dataset, range(0, len(dataset), sample_stride))
    return dataset


def check_rows(acts, dataset, cfg):
    if acts.shape[0] != len(dataset):
        raise ValueError(f"{cfg.activations} has {acts.shape[0]} rows but {cfg.image_dir} with "
                         f"sample_stride={cfg.sample_stride} has {len(dataset)} images")


@torch.no_grad()
def encode_dataset(encode_fn, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return torch.cat([encode_fn(images.to(device)).float().cpu()
                      for images, _ in tqdm(loader, desc="image features")])


def cached(cache_dir, name, key, compute_fn):
    """Return ``compute_fn()``, cached under ``cache_dir`` by a hash of ``key`` (a JSON-able dict)."""
    if cache_dir is None:
        return compute_fn()
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{name}_{digest}.pt")
    if os.path.exists(path):
        print(f"  loading cached {name} from {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    value = compute_fn()
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(value, path)
    print(f"  cached {name} to {path}")
    return value


def images_key(cfg, n_images):
    return {"image_dir": os.path.abspath(cfg.image_dir), "sample_stride": cfg.sample_stride,
            "n_images": n_images}


def words_digest(words):
    return hashlib.sha1("\n".join(words).encode()).hexdigest()


def save_descriptions(similarities, words, cfg):
    """Best concept per unit -> ``output_dir/descriptions.csv`` (+ ``config.json``)."""
    values, ids = similarities.max(dim=1)
    df = pd.DataFrame({
        "layer": [cfg.layer_name] * len(ids),
        "unit": range(len(ids)),
        "description": [words[i] for i in ids.tolist()],
        "similarity": values.cpu().numpy(),
    })
    os.makedirs(cfg.output_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg.output_dir, "descriptions.csv"), index=False)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        f.write(config_to_json(cfg))
    print(f"Saved descriptions of {len(df)} units to {cfg.output_dir}")
