"""
PartImageNet annotations and pre-extracted features for the composition evaluation.

- Ontology: a COCO-style JSON file, ``{"categories": [{"id", "name",
  "supercategory"}, ...]}``. Part ids must be ``0 .. n_parts - 1``;
  supercategories are indexed in sorted name order.
- Label maps: one PNG per image (``<image_id>.png``) whose pixel values are part
  ids or the background label. They are downsampled to the backbone's patch
  grid with nearest-neighbour sampling, picking the same source pixels as
  ``cv2.resize(..., interpolation=cv2.INTER_NEAREST)``.
- Features: ``(N, T, d_local)`` patch tokens and ``(N, d_global)`` global
  features in the order of an ``image_ids`` JSON list (see
  ``extract_partimagenet_features.py``).
"""

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

BACKGROUND = -1  # label of non-part grid cells after loading


@dataclass(frozen=True)
class Ontology:
    part_names: list
    supercat_names: list
    part_to_supercat: np.ndarray  # (n_parts,) supercategory index of every part

    @property
    def n_parts(self):
        return len(self.part_names)

    @property
    def n_supercats(self):
        return len(self.supercat_names)

    def ground_truth(self):
        """Binary ``(n_supercats, n_parts)`` matrix: ``C_gt[g, p] = 1`` iff part p belongs to g."""
        C_gt = np.zeros((self.n_supercats, self.n_parts), dtype=np.int8)
        C_gt[self.part_to_supercat, np.arange(self.n_parts)] = 1
        return C_gt


def load_ontology(path):
    with open(path) as f:
        categories = json.load(f)["categories"]
    ids = sorted(int(c["id"]) for c in categories)
    if ids != list(range(len(categories))):
        raise ValueError(f"{path}: part ids must be 0..{len(categories) - 1}, got {ids}")
    by_id = {int(c["id"]): c for c in categories}
    supercat_names = sorted({c["supercategory"] for c in categories})
    supercat_index = {name: g for g, name in enumerate(supercat_names)}
    return Ontology(
        part_names=[by_id[p]["name"] for p in ids],
        supercat_names=supercat_names,
        part_to_supercat=np.array([supercat_index[by_id[p]["supercategory"]] for p in ids],
                                  dtype=np.int64),
    )


def patch_grid_size(n_tokens):
    """Side of the square patch grid holding ``n_tokens`` tokens."""
    grid = math.isqrt(n_tokens)
    if grid * grid != n_tokens:
        raise ValueError(f"{n_tokens} patch tokens do not form a square grid")
    return grid


def _nearest_source_index(n_src, n_dst):
    # Same arithmetic as OpenCV's resizeNN: floor(i * (1 / (n_dst / n_src))).
    scale = 1.0 / (n_dst / n_src)
    return np.minimum(np.floor(np.arange(n_dst) * scale).astype(np.int64), n_src - 1)


def downsample_label_map(label_map, grid):
    """Nearest-neighbour downsampling of an ``(H, W)`` label map to ``(grid, grid)``."""
    h, w = label_map.shape
    return label_map[np.ix_(_nearest_source_index(h, grid), _nearest_source_index(w, grid))]


def load_grid_labels(image_ids, labelmap_dir, grid, n_parts, background_label):
    """
    Part label of every patch, ``(N, grid * grid)`` int64 in raster order.

    Part ids are kept; background cells become ``BACKGROUND``. Any other value
    raises, so a wrong ``background_label`` or ontology cannot pass silently.
    """
    labels = np.empty((len(image_ids), grid * grid), dtype=np.int64)
    for i, image_id in enumerate(image_ids):
        label_map = np.array(Image.open(os.path.join(labelmap_dir, f"{image_id}.png")))
        cells = downsample_label_map(label_map, grid).reshape(-1).astype(np.int64)
        unknown = (cells != background_label) & ((cells < 0) | (cells >= n_parts))
        if unknown.any():
            raise ValueError(f"{image_id}: unexpected labels {sorted(set(cells[unknown].tolist()))}")
        labels[i] = np.where(cells == background_label, BACKGROUND, cells)
    return labels


def image_supercategories(labels, part_to_supercat):
    """
    Supercategory of every image: the majority vote of its part cells, ties
    going to the supercategory seen first in raster order. ``-1`` for images
    without part cells.
    """
    out = np.full(labels.shape[0], -1, dtype=np.int64)
    n_supercats = int(part_to_supercat.max()) + 1
    for i, row in enumerate(labels):
        votes = part_to_supercat[row[row != BACKGROUND]]
        if votes.size == 0:
            continue
        counts = np.bincount(votes, minlength=n_supercats)
        winners = np.flatnonzero(counts == counts.max())
        out[i] = votes[np.isin(votes, winners)][0]
    return out


def load_features(patch_tokens_path, global_features_path, image_ids_path):
    """Memory-mapped ``(N, T, d_local)`` patch tokens, ``(N, d_global)`` global features and ids."""
    patch_tokens = torch.load(patch_tokens_path, map_location="cpu", weights_only=True, mmap=True)
    global_features = torch.load(global_features_path, map_location="cpu", weights_only=True, mmap=True)
    with open(image_ids_path) as f:
        image_ids = json.load(f)
    if patch_tokens.dim() != 3 or global_features.dim() != 2:
        raise ValueError(f"Expected (N, T, d) patch tokens and (N, d) global features, got "
                         f"{tuple(patch_tokens.shape)} and {tuple(global_features.shape)}")
    if not patch_tokens.shape[0] == global_features.shape[0] == len(image_ids):
        raise ValueError(f"Row counts differ: {patch_tokens.shape[0]} patch, "
                         f"{global_features.shape[0]} global, {len(image_ids)} image ids")
    return patch_tokens, global_features, image_ids
