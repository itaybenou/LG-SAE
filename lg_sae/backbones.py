"""
Feature extractors for the supported vision backbones.

Every extractor exposes ``preprocess`` (a PIL -> tensor transform) and
``extract(images) -> (global_features, patch_tokens)`` with shapes
``(B, d_global)`` and ``(B, T, d_local)``.

- CLIP: patch tokens are the raw residual stream after ``token_layer``
  transformer blocks (CLS dropped); the global feature is the final projected
  image embedding (``encode_image``).
- DINOv2: patch tokens and CLS token are both read from the raw residual
  stream after ``token_layer`` blocks (before the final LayerNorm; register
  tokens are dropped).
"""

import os

import torch
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _check_layer(token_layer, n_blocks):
    if not 1 <= token_layer <= n_blocks:
        raise ValueError(f"token_layer must be in [1, {n_blocks}], got {token_layer}")


class CLIPExtractor:
    def __init__(self, model_name, token_layer, device):
        from .third_party import clip

        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device
        _check_layer(token_layer, len(self.model.visual.transformer.resblocks))
        self.token_layer = token_layer

    @torch.no_grad()
    def extract(self, images):
        visual = self.model.visual
        x = visual.conv1(images.type(self.model.dtype))          # (B, width, grid, grid)
        x = x.flatten(2).permute(0, 2, 1)                         # (B, grid**2, width)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)                     # (L, B, width)

        patch_tokens = None
        for i, block in enumerate(visual.transformer.resblocks, start=1):
            x = block(x)
            if i == self.token_layer:
                patch_tokens = x.permute(1, 0, 2)[:, 1:, :]
        image_embedding = visual.ln_post(x.permute(1, 0, 2)[:, 0, :])
        if visual.proj is not None:
            image_embedding = image_embedding @ visual.proj
        return image_embedding, patch_tokens


class DINOv2Extractor:
    def __init__(self, model_name, token_layer, hub_repo, image_size, device):
        hub_repo = os.path.expanduser(hub_repo)
        source = "local" if os.path.isdir(hub_repo) else "github"
        self.model = torch.hub.load(hub_repo, model_name, source=source).eval().to(device)
        self.device = device
        _check_layer(token_layer, len(self.model.blocks))
        self.token_layer = token_layer
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def extract(self, images):
        x = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks[:self.token_layer]:
            x = block(x)
        return x[:, 0], x[:, 1 + self.model.num_register_tokens:]


def build_extractor(backbone, model_name, token_layer, device, hub_repo=None, image_size=None):
    if backbone == "clip":
        return CLIPExtractor(model_name, token_layer, device)
    if backbone == "dinov2":
        if hub_repo is None or image_size is None:
            raise ValueError("hub_repo and image_size are required for the dinov2 backbone")
        return DINOv2Extractor(model_name, token_layer, hub_repo, image_size, device)
    raise ValueError(f"Unknown backbone {backbone!r}; expected 'clip' or 'dinov2'")
