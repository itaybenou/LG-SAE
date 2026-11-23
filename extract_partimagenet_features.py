"""
Extract backbone features for a flat directory of images (e.g. PartImageNet test).

Uses the same extractors as save_shards.py, so the features match the SAE
training data. Images are processed in sorted file-name order; the image id is
the file name without extension.

Output
  <output_dir>/patch_tokens.pt  (N, T, d_local)  patch tokens
  <output_dir>/global.pt        (N, d_global)    CLIP image embedding / DINOv2 CLS token
  <output_dir>/image_ids.json   [N]              image id of every row
  <output_dir>/manifest.json

Example
  python extract_partimagenet_features.py --config configs/partimagenet/clip_vitb16.yaml \\
      --image_dir /data/PartImageNet/images/test --output_dir /data/partimagenet/clip_vitb16
"""

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
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
    Option("image_dir", str, "Flat directory of images."),
    Option("output_dir", str, "Directory that receives the feature files and manifest."),
    Option("batch_size", int, "Inference batch size."),
    Option("num_workers", int, "DataLoader workers."),
    Option("device", str, "Torch device, e.g. cuda:0."),
]


class FlatImageDataset(Dataset):
    """Images of a flat directory in sorted file-name order."""

    def __init__(self, image_dir, transform):
        self.files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTENSIONS))
        if not self.files:
            raise FileNotFoundError(f"No images in {image_dir}")
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = [os.path.splitext(f)[0] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(default_loader(os.path.join(self.image_dir, self.files[index])))


def main():
    cfg = load_config(OPTIONS, description=__doc__)

    extractor = build_extractor(cfg.backbone, cfg.model_name, cfg.token_layer, cfg.device,
                                hub_repo=cfg.hub_repo, image_size=cfg.image_size)
    dataset = FlatImageDataset(cfg.image_dir, extractor.preprocess)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    print(f"{cfg.backbone} {cfg.model_name} (token_layer={cfg.token_layer}) on {cfg.device}: "
          f"{len(dataset)} images from {cfg.image_dir}")

    global_features, patch_tokens = [], []
    for images in tqdm(loader, desc="extracting"):
        g, p = extractor.extract(images.to(cfg.device, non_blocking=True))
        global_features.append(g.cpu())
        patch_tokens.append(p.cpu())
    global_features = torch.cat(global_features)
    patch_tokens = torch.cat(patch_tokens)

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(patch_tokens, os.path.join(cfg.output_dir, "patch_tokens.pt"))
    torch.save(global_features, os.path.join(cfg.output_dir, "global.pt"))
    with open(os.path.join(cfg.output_dir, "image_ids.json"), "w") as f:
        json.dump(dataset.image_ids, f)
    manifest = {
        "total_samples": len(dataset),
        "shapes": {"patch_tokens": list(patch_tokens.shape[1:]), "global": list(global_features.shape[1:])},
        "dtypes": {"patch_tokens": str(patch_tokens.dtype).replace("torch.", ""),
                   "global": str(global_features.dtype).replace("torch.", "")},
        "config": vars(cfg),
    }
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done: patch_tokens {tuple(patch_tokens.shape)}, global {tuple(global_features.shape)} "
          f"-> {cfg.output_dir}")


if __name__ == "__main__":
    main()
