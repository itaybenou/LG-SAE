"""
CLIP-DINOiser dense image features (Wysoczanska et al., ECCV 2024) on top of open_clip.

Re-implements the dense output of the official model used by spatial
CLIP-Dissect, without its mmcv / mmseg / hydra dependencies:

1. MaskCLIP features: the last ViT block is applied with its attention
   replaced by the value path (no token mixing), followed by ``ln_post`` and
   the CLIP projection, giving one CLIP-space embedding per patch.
2. CLIP-DINOiser refinement: the trained ``obj_proj`` head maps intermediate
   patch features to embeddings whose correlations (thresholded at
   ``gamma``) weight an average of the MaskCLIP features over all patches.
"""

import open_clip
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from .prompt_templates import IMAGENET_TEMPLATES

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
# The released obj_proj head is trained on the MLP input (ln_2 output) of the third-to-last block.
CORR_FEATS_BLOCK = -3


def dense_preprocess(image_size):
    """Image transform for ``CLIPDINOiser.encode_images`` (normalization happens inside the model)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


class CLIPDINOiser(nn.Module):
    def __init__(self, checkpoint_path, clip_model, clip_pretrained, gamma, device):
        super().__init__()
        clip, _ = open_clip.create_model_from_pretrained(clip_model, pretrained=clip_pretrained)
        self.clip = clip
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.gamma = gamma
        self.patch_size = self.clip.visual.patch_size[0]
        self.normalize = transforms.Normalize(OPENAI_MEAN, OPENAI_STD)

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"]
        weight = state["obj_proj.weight"]
        self.obj_proj = nn.Conv2d(weight.shape[1], weight.shape[0], weight.shape[2],
                                  padding=weight.shape[2] // 2, padding_mode="replicate")
        self.obj_proj.load_state_dict({"weight": weight, "bias": state["obj_proj.bias"]})

        blocks = self.clip.visual.transformer.resblocks
        self._hooked = {}
        blocks[-2].register_forward_hook(lambda m, i, out: self._hooked.__setitem__("last_in", out))
        blocks[CORR_FEATS_BLOCK].ln_2.register_forward_hook(
            lambda m, i, out: self._hooked.__setitem__("corr_in", out))

        self.device = device
        self.to(device).eval()

    def grid_size(self, image_size):
        if image_size % self.patch_size:
            raise ValueError(f"image_size={image_size} is not a multiple of the patch size {self.patch_size}")
        return image_size // self.patch_size, image_size // self.patch_size

    @torch.no_grad()
    def encode_images(self, images):
        """``(B, 3, S, S)`` images in ``[0, 1]`` -> ``(B, C, S / patch, S / patch)`` dense CLIP features."""
        visual = self.clip.visual
        b = images.shape[0]
        h, w = self.grid_size(images.shape[-1])
        pos_embed = visual.positional_embedding.data
        if (h, w) != tuple(visual.grid_size):
            visual.positional_embedding.data = self._resize_pos_embed(pos_embed, (h, w))
        try:
            visual(self.normalize(images))
        finally:
            visual.positional_embedding.data = pos_embed

        # MaskCLIP: value path of the last block, then ln_post + projection (sequence-first tokens).
        tokens = self._value_path(self._hooked["last_in"], visual.transformer.resblocks[-1])
        tokens = visual.ln_post(tokens.permute(1, 0, 2))[:, 1:] @ visual.proj
        clip_feats = tokens.transpose(1, 2).reshape(b, -1, h, w)

        corr_in = self._hooked["corr_in"].permute(1, 2, 0)[:, :, 1:].reshape(b, -1, h, w)
        proj = F.normalize(self.obj_proj(corr_in).flatten(2), dim=1)
        corrs = proj.transpose(1, 2) @ proj
        if self.gamma > 0:
            corrs = corrs.masked_fill(corrs < self.gamma, 0.0)
        pooled = clip_feats.flatten(2) @ corrs.transpose(1, 2) / (corrs.sum(dim=-1)[:, None] + 1e-6)
        return pooled.reshape(clip_feats.shape)

    @torch.no_grad()
    def encode_text(self, words, batch_size):
        """Template-averaged, L2-normalized concept embeddings ``(K, C)`` (MaskCLIP text head)."""
        n_templates = len(IMAGENET_TEMPLATES)
        words_per_batch = max(1, batch_size // n_templates)
        out = []
        for start in tqdm(range(0, len(words), words_per_batch), desc="text features"):
            chunk = words[start:start + words_per_batch]
            prompts = [template.format(word) for word in chunk for template in IMAGENET_TEMPLATES]
            emb = F.normalize(self.clip.encode_text(self.tokenizer(prompts).to(self.device)).float(), dim=-1)
            out.append(F.normalize(emb.reshape(len(chunk), n_templates, -1).mean(dim=1), dim=-1).cpu())
        return torch.cat(out)

    @staticmethod
    def _value_path(x, block):
        """``block`` with self-attention replaced by its value + output projections."""
        dim = x.shape[-1]
        attn = block.attn
        v = F.linear(block.ln_1(x), attn.in_proj_weight[2 * dim:], attn.in_proj_bias[2 * dim:])
        v = F.linear(v, attn.out_proj.weight, attn.out_proj.bias) + x
        return v + block.mlp(block.ln_2(v))

    def _resize_pos_embed(self, pos_embed, grid):
        gh, gw = self.clip.visual.grid_size
        cls_pos, patch_pos = pos_embed[:1], pos_embed[1:]
        patch_pos = patch_pos.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=grid, mode="bicubic", align_corners=False)
        return torch.cat([cls_pos, patch_pos.flatten(2)[0].T])
