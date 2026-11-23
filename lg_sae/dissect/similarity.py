"""
Soft-WPMI concept scoring from CLIP-Dissect (Oikarinen & Weng, ICLR 2023).

One implementation serves both the standard (image-level) and the spatial
(location-level) variant: the probing "rows" are either images or image
locations, and CLIP probabilities are only computed at the rows where each
unit activates most, so the full ``(rows, concepts)`` matrix is never built.
"""

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def top_activating_indices(activations, top_k, device, unit_batch_size):
    """
    Rows where each unit activates most, sorted from highest to lowest.

    ``activations``: ``(N, U)`` or ``(N, U, H, W)`` (any float dtype, usually on CPU).
    Spatial locations are flattened as ``n * H * W + y * W + x``.
    Returns ``(top_k, U)`` int64 indices on CPU.
    """
    acts = activations if activations.dim() == 4 else activations[:, :, None, None]
    n, n_units, h, w = acts.shape
    if top_k > n * h * w:
        raise ValueError(f"top_k={top_k} exceeds the number of probing rows ({n * h * w})")
    indices = []
    for start in tqdm(range(0, n_units, unit_batch_size), desc="top activations"):
        chunk = acts[:, start:start + unit_batch_size].to(device).float()
        chunk = chunk.permute(1, 0, 2, 3).reshape(chunk.shape[1], n * h * w)
        indices.append(chunk.topk(top_k, dim=1).indices.T.cpu())
    return torch.cat(indices, dim=1)


@torch.no_grad()
def soft_wpmi(image_feats, text_feats, top_inds, *, softmax_temp, lam, p_start, p_end,
              min_prob, device, unit_batch_size):
    """
    Soft-WPMI score of every (unit, concept) pair.

    ``image_feats``: ``(M, C)`` CLIP image features of the probing rows.
    ``text_feats``:  ``(K, C)`` CLIP text features of the concepts.
    ``top_inds``:    ``(top_k, U)`` from ``top_activating_indices``.

    With ``P = softmax(softmax_temp * cos(image, text))`` over concepts, the
    ``i``-th top row of a unit counts as an example with probability ``p_i``
    (linearly decreasing from ``p_start`` to ``p_end``), and
    ``score = log P(concept | unit examples) - lam * log P(concept)``.
    Returns ``(U, K)`` scores on ``device``.
    """
    image_feats = F.normalize(image_feats.float(), dim=-1).to(device)
    text_feats = F.normalize(text_feats.float(), dim=-1).to(device)
    top_k = top_inds.shape[0]
    p_example = p_start - torch.arange(top_k, device=device) / top_k * (p_start - p_end)
    p_example = p_example[None, :, None]

    log_p_given_unit = []
    for start in tqdm(range(0, top_inds.shape[1], unit_batch_size), desc="soft-WPMI"):
        inds = top_inds[:, start:start + unit_batch_size].T.to(device)
        probs = torch.softmax(softmax_temp * (image_feats[inds.reshape(-1)] @ text_feats.T), dim=-1)
        probs = probs.reshape(*inds.shape, -1)
        log_p_given_unit.append(torch.log(1 + p_example * (probs - 1) + min_prob).sum(dim=1))
    log_p_given_unit = torch.cat(log_p_given_unit)

    log_p = torch.logsumexp(log_p_given_unit, dim=0, keepdim=True) - math.log(log_p_given_unit.shape[0])
    return log_p_given_unit - lam * log_p
