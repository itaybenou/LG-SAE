"""
Composition (ontology) evaluation of an LG-SAE bridge on part-annotated images.

Every ground-truth supercategory g is matched to a global concept i*(g) and
every part p to a local concept j*(p); the bridge predicts the edge (g, p) iff
``C[i*(g), j*(p)] > tau``, and the predictions are scored against the binary
ground-truth ontology ``C_gt``.

- i*(g): the global concept whose code has the highest point-biserial
  correlation with the image-level indicator [supercategory == g].
- j*(p), the canonical selection with a frequency cap:
    1. local concepts active on more than ``freq_cap`` of the tokens of a
       reference set are excluded (their codes are zeroed);
    2. every token keeps only its ``token_top_m`` largest codes;
    3. for every image containing p, the cosine between each concept's code
       map and the binary mask of p, averaged over those images
       -> ``mean_cos[p, c]``;
    4. concepts with fewer than ``min_active_patches`` active tokens (after
       step 2) are excluded;
    5. ``score[p, c] = mean_cos[p, c] - mean_{p' != p} mean_cos[p', c]``
       over the other parts present in the data;
    6. j*(p) = argmax_c score[p, c].
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .partimagenet import BACKGROUND


def _batches(n, batch_size, desc):
    return tqdm(range(0, n, batch_size), desc=desc, leave=False)


@torch.no_grad()
def encode_global(model, features, batch_size, device):
    """Global codes ``(N, n_global)`` on the CPU."""
    codes = [model.encode_global(features[s:s + batch_size].to(device).float())[1].float().cpu()
             for s in _batches(features.shape[0], batch_size, "global codes")]
    return torch.cat(codes)


@torch.no_grad()
def local_token_frequency(model, patch_tokens, batch_size, device):
    """Fraction of the tokens of ``patch_tokens`` (N, T, d) on which each local concept is active."""
    n, n_tokens = patch_tokens.shape[:2]
    active = torch.zeros(model.n_local, dtype=torch.float64, device=device)
    for s in _batches(n, batch_size, "token frequency"):
        _, z = model.encode_local(patch_tokens[s:s + batch_size].to(device).float())
        active += (z > 0).sum(dim=0).double()
    return (active / (n * n_tokens)).cpu()


def select_i_star(z_global, supercats, n_supercats, eps):
    """
    Point-biserial selection of one global concept per supercategory::

        r_pb(i, g) = (mu_g(i) - mu_not_g(i)) / (std(i) + eps) * sqrt(n_g * n_not_g / N^2)

    Images with ``supercats == -1`` only count towards "not g". Returns
    ``(i_star, valid)``; supercategories with no (or only) images are invalid.
    """
    n = z_global.shape[0]
    std = z_global.std(dim=0)
    i_star = np.full(n_supercats, -1, dtype=np.int64)
    valid = np.zeros(n_supercats, dtype=bool)
    for g in range(n_supercats):
        in_g = torch.from_numpy(supercats == g)
        n_g = int(in_g.sum())
        if n_g in (0, n):
            continue
        prevalence = n_g * (n - n_g) / n ** 2
        r_pb = (z_global[in_g].mean(dim=0) - z_global[~in_g].mean(dim=0)) / (std + eps) * prevalence ** 0.5
        i_star[g] = int(np.argmax(r_pb.numpy()))
        valid[g] = True
    return i_star, valid


def mask_token_top_m(z, m):
    """Keep the ``m`` largest codes of every token of ``z`` (..., n_local)."""
    top = torch.topk(z, m, dim=-1)
    return torch.zeros_like(z).scatter(-1, top.indices, top.values)


@torch.no_grad()
def part_cosine(model, patch_tokens, labels, n_parts, excluded, token_top_m, eps, batch_size, device):
    """
    Steps 1-3 of the j* selection.

    ``labels`` is ``(N, T)`` (part id or ``BACKGROUND``), ``excluded`` a boolean
    ``(n_local,)`` mask of frequency-capped concepts. Returns ``mean_cos``
    ``(n_parts, n_local)`` (NaN rows for parts absent from the data) and the
    number of active tokens of every concept after the top-m mask.
    """
    n, n_tokens = labels.shape
    excluded = excluded.to(device)
    labels = torch.from_numpy(labels)
    cos_sum = torch.zeros(n_parts, model.n_local, dtype=torch.float64, device=device)
    n_images = torch.zeros(n_parts, dtype=torch.float64, device=device)
    active = torch.zeros(model.n_local, dtype=torch.float64, device=device)

    for s in _batches(n, batch_size, "part cosine"):
        _, z = model.encode_local(patch_tokens[s:s + batch_size].to(device).float())
        z = mask_token_top_m(z.float().masked_fill(excluded, 0.0), token_top_m)
        z = z.reshape(-1, n_tokens, model.n_local)                       # (B, T, n_local)

        lab = labels[s:s + batch_size].to(device)
        is_part = lab != BACKGROUND
        masks = torch.zeros(*lab.shape, n_parts, device=device)          # (B, T, P)
        masks[is_part] = F.one_hot(lab[is_part], n_parts).float()
        mask_size = masks.sum(dim=1)                                     # (B, P)
        present = mask_size > 0

        dot = torch.bmm(masks.transpose(1, 2), z)                       # (B, P, n_local)
        z_norm = z.square().sum(dim=1).sqrt()                            # (B, n_local)
        cos = dot / (z_norm[:, None, :] * mask_size.sqrt()[:, :, None] + eps)
        cos_sum += torch.einsum("bp,bpc->pc", present.double(), cos.double())
        n_images += present.sum(dim=0).double()
        active += (z > 0).sum(dim=(0, 1)).double()

    mean_cos = cos_sum / n_images[:, None]                               # NaN for absent parts
    return mean_cos.cpu().numpy(), active.cpu().numpy()


def select_j_star(mean_cos, active_tokens, min_active_patches):
    """
    Steps 4-6 of the j* selection. Returns ``(j_star, valid)``; parts absent
    from the data are invalid.
    """
    n_parts = mean_cos.shape[0]
    valid = ~np.isnan(mean_cos).any(axis=1)
    eligible = active_tokens >= min_active_patches
    if valid.sum() < 2 or not eligible.any():
        raise ValueError(f"j* needs >= 2 parts present ({valid.sum()}) and >= 1 concept with "
                         f">= {min_active_patches} active tokens ({eligible.sum()})")
    M = mean_cos[valid]                                                  # (R, n_local)
    others = (M.sum(axis=0, keepdims=True) - M) / (M.shape[0] - 1)
    score = np.where(eligible, M - others, -np.inf)
    j_star = np.full(n_parts, -1, dtype=np.int64)
    j_star[valid] = score.argmax(axis=1)
    return j_star, valid


def edge_metrics(C, i_star, supercat_valid, j_star, part_valid, C_gt, tau):
    """
    Precision / recall / F1 of the predicted edges ``C[i*(g), j*(p)] > tau``
    against ``C_gt`` over all pairs with a valid i*(g) and j*(p).
    """
    pairs = np.outer(supercat_valid, part_valid)
    predicted = C[np.ix_(np.maximum(i_star, 0), np.maximum(j_star, 0))] > tau
    truth = C_gt.astype(bool)
    tp = int((predicted & truth & pairs).sum())
    fp = int((predicted & ~truth & pairs).sum())
    fn = int((~predicted & truth & pairs).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "n_skipped_pairs": int((~pairs).sum())}
