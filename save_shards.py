"""
Extract backbone features for an ImageFolder dataset and save them as shards.

Output layout (one file per shard in each sub-directory, matched by name):

  <output_dir>/global/shard_00000.pt        (n, d_global)  CLIP image embedding / DINOv2 CLS token
  <output_dir>/patch_tokens/shard_00000.pt  (n, T, d_local) patch tokens (if save_patch_tokens)
  <output_dir>/idxs/shard_00000.pt          (n,)  dataset index of every row
  <output_dir>/labels/shard_00000.pt        (n,)  class label of every row
  <output_dir>/manifest.json

With --shuffle true, images are visited in a seeded random order so that the
rows of a shard are not consecutive; otherwise in dataset order.

Example
  python save_shards.py --config configs/shards/dinov2_vitb14.yaml \\
      --image_dir /data/imagenet/train --output_dir /data/dinov2_vitb14/train --shuffle true
"""

import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from lg_sae.backbones import build_extractor
from lg_sae.config import Option, load_config

OPTIONS = [
    Option("backbone", str, "Feature extractor.", choices=("clip", "dinov2")),
    Option("model_name", str, "Backbone variant, e.g. ViT-B/16 (CLIP) or dinov2_vitb14 (DINOv2)."),
    Option("token_layer", int, "Read patch tokens (and the DINOv2 CLS token) after this many "
                               "transformer blocks."),
    Option("hub_repo", str, "DINOv2 torch.hub repo: a local checkout path or 'facebookresearch/dinov2'.",
           nullable=True),
    Option("image_size", int, "DINOv2 input resolution (CLIP uses its own preprocessing).", nullable=True),
    Option("image_dir", str, "ImageFolder root (one sub-directory per class)."),
    Option("output_dir", str, "Directory that receives the shard sub-directories and manifest."),
    Option("shuffle", bool, "Visit images in a seeded random order."),
    Option("seed", int, "Seed of the shuffling order."),
    Option("shard_size", int, "Images per shard (the last shard holds the remainder)."),
    Option("save_patch_tokens", bool, "Also save patch tokens (otherwise only global features)."),
    Option("batch_size", int, "Inference batch size."),
    Option("num_workers", int, "DataLoader workers."),
    Option("device", str, "Torch device, e.g. cuda:0."),
]


class IndexedImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns the sample index."""

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index


class ShardWriter:
    """Buffers rows of several aligned fields and writes fixed-size shards."""

    def __init__(self, output_dir, fields, shard_size):
        self.output_dir = output_dir
        self.fields = fields
        self.shard_size = shard_size
        self.buffers = {f: [] for f in fields}
        self.n_buffered = 0
        self.n_shards = 0
        self.n_written = 0
        self.shapes = {}
        self.dtypes = {}
        for field in fields:
            path = os.path.join(output_dir, field)
            os.makedirs(path, exist_ok=True)
            if any(f.endswith(".pt") for f in os.listdir(path)):
                raise FileExistsError(f"{path} already contains shards; use an empty output_dir")

    def add(self, **rows):
        for field in self.fields:
            self.buffers[field].append(rows[field])
        self.n_buffered += rows[self.fields[0]].shape[0]
        while self.n_buffered >= self.shard_size:
            self._write(self.shard_size)

    def close(self):
        if self.n_buffered > 0:
            self._write(self.n_buffered)

    def _write(self, n):
        for field in self.fields:
            merged = torch.cat(self.buffers[field], dim=0)
            shard, rest = merged[:n], merged[n:]
            if rest.shape[0] > 0:
                # torch.save would otherwise serialize the whole underlying storage.
                shard, rest = shard.clone(), rest.clone()
            self.buffers[field] = [rest] if rest.shape[0] > 0 else []
            torch.save(shard, os.path.join(self.output_dir, field, f"shard_{self.n_shards:05d}.pt"))
            self.shapes[field] = list(shard.shape[1:])
            self.dtypes[field] = str(shard.dtype).replace("torch.", "")
        self.n_buffered -= n
        self.n_written += n
        self.n_shards += 1
        print(f"  wrote shard {self.n_shards - 1:05d} ({n} images, {self.n_written} total)", flush=True)


def main():
    cfg = load_config(OPTIONS, description=__doc__)

    extractor = build_extractor(cfg.backbone, cfg.model_name, cfg.token_layer, cfg.device,
                                hub_repo=cfg.hub_repo, image_size=cfg.image_size)
    dataset = IndexedImageFolder(cfg.image_dir, transform=extractor.preprocess)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                        generator=torch.Generator().manual_seed(cfg.seed),
                        num_workers=cfg.num_workers, pin_memory=True)

    fields = ["global", "patch_tokens", "idxs", "labels"] if cfg.save_patch_tokens \
        else ["global", "idxs", "labels"]
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = ShardWriter(cfg.output_dir, fields, cfg.shard_size)

    print(f"{cfg.backbone} {cfg.model_name} (token_layer={cfg.token_layer}) on {cfg.device}")
    print(f"{len(dataset)} images in {len(dataset.classes)} classes from {cfg.image_dir}")
    print(f"shuffle={cfg.shuffle} shard_size={cfg.shard_size} -> {cfg.output_dir}/{{{','.join(fields)}}}")

    for images, labels, indices in tqdm(loader, desc="extracting"):
        global_features, patch_tokens = extractor.extract(images.to(cfg.device, non_blocking=True))
        rows = {"global": global_features.cpu(), "idxs": indices, "labels": labels}
        if cfg.save_patch_tokens:
            rows["patch_tokens"] = patch_tokens.cpu()
        writer.add(**rows)
    writer.close()

    manifest = {
        "total_samples": writer.n_written,
        "num_shards": writer.n_shards,
        "shapes": writer.shapes,
        "dtypes": writer.dtypes,
        "classes": dataset.classes,
        "config": vars(cfg),
    }
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done: {writer.n_written} images in {writer.n_shards} shards")


if __name__ == "__main__":
    main()
