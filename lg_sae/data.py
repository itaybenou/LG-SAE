"""
Streaming over pre-extracted backbone features.

Features are stored as ``.pt`` / ``.pth`` tensors, either as one file or as a
directory of shards (see ``save_shards.py``):

- global features: ``(N, d_global)`` per file
- patch tokens:    ``(N, T, d_local)`` per file

In directory mode, global and patch shards are paired by file stem
(``global/shard_00003.pt`` <-> ``patch_tokens/shard_00003.pt``).
"""

import os
import random
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset

SHARD_EXTENSIONS = (".pt", ".pth")
MODES = ("global", "patch", "paired")


def list_shards(path):
    """Return ``[path]`` for a file, or the sorted shard files of a directory."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    files = sorted(os.path.join(path, f) for f in os.listdir(path)
                   if f.lower().endswith(SHARD_EXTENSIONS))
    if not files:
        raise FileNotFoundError(f"No {SHARD_EXTENSIONS} files in {path}")
    return files


def pair_shards(global_path, patch_path):
    """``[(global_file, patch_file), ...]`` matched by stem (or the two files themselves)."""
    if os.path.isfile(global_path) and os.path.isfile(patch_path):
        return [(global_path, patch_path)]
    if not (os.path.isdir(global_path) and os.path.isdir(patch_path)):
        raise ValueError("global_dir and patch_dir must both be files or both be directories")
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    patch_by_stem = {stem(f): f for f in list_shards(patch_path)}
    global_by_stem = {stem(f): f for f in list_shards(global_path)}
    if global_by_stem.keys() != patch_by_stem.keys():
        only_g = sorted(global_by_stem.keys() - patch_by_stem.keys())
        only_p = sorted(patch_by_stem.keys() - global_by_stem.keys())
        raise ValueError(f"Unpaired shards: global-only {only_g[:5]}, patch-only {only_p[:5]}")
    return [(global_by_stem[s], patch_by_stem[s]) for s in sorted(global_by_stem)]


def _load(path, mmap=False):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


@dataclass
class ShardInfo:
    n_samples: int
    n_patches: int
    global_dim: int
    local_dim: int
    n_shards: int


def inspect_shards(global_path, patch_path):
    """Infer dataset size and feature shapes (memory-mapped, so no full loads)."""
    pairs = pair_shards(global_path, patch_path)
    n_samples = 0
    for g_file, p_file in pairs:
        g, p = _load(g_file, mmap=True), _load(p_file, mmap=True)
        if g.dim() != 2 or p.dim() != 3 or g.shape[0] != p.shape[0]:
            raise ValueError(f"Expected (N, d) and (N, T, d) tensors, got {tuple(g.shape)} in "
                             f"{g_file} and {tuple(p.shape)} in {p_file}")
        n_samples += g.shape[0]
    return ShardInfo(n_samples=n_samples, n_patches=p.shape[1], global_dim=g.shape[1],
                     local_dim=p.shape[2], n_shards=len(pairs))


class ShardDataset(IterableDataset):
    """
    Yields ready-made float32 batches from sharded features.

    Each epoch visits the shards in (optionally shuffled) order, loads one
    shard at a time and yields full batches from a (optionally shuffled)
    permutation of its rows; the last incomplete batch of each shard is
    dropped. Shuffling uses Python's and torch's global RNGs.

    Modes
    -----
    ``global``: ``(B, d_global)`` image-level features.
    ``patch``:  ``(B, d_local)`` individual patch tokens (patches are flattened).
    ``paired``: ``((B, d_global), (B, T, d_local))`` aligned image/patch features.
    """

    def __init__(self, global_path, patch_path, mode, batch_size, shuffle=True):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        if mode == "global":
            self.shards = [(f,) for f in list_shards(global_path)]
        elif mode == "patch":
            self.shards = [(f,) for f in list_shards(patch_path)]
        else:
            self.shards = pair_shards(global_path, patch_path)
        self._cached = None

    def _load_shard(self, files):
        if self._cached is not None and self._cached[0] == files:
            return self._cached[1]
        # Kept in the stored dtype (often fp16); batches are cast to float32 on the fly.
        tensors = [_load(f) for f in files]
        if self.mode == "patch":
            tensors = [tensors[0].reshape(-1, tensors[0].shape[-1])]
        if len(self.shards) == 1:
            self._cached = (files, tensors)
        return tensors

    def __iter__(self):
        order = list(self.shards)
        if self.shuffle:
            random.shuffle(order)
        for files in order:
            tensors = None  # release the previous shard before loading the next one
            tensors = self._load_shard(files)
            n = tensors[0].shape[0]
            idx = torch.randperm(n) if self.shuffle else torch.arange(n)
            for start in range(0, n - self.batch_size + 1, self.batch_size):
                rows = idx[start:start + self.batch_size]
                batch = [t[rows].float() for t in tensors]
                yield batch[0] if len(batch) == 1 else tuple(batch)
