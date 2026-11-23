"""Training metrics: R^2, L0, dead latents and running averages."""

from collections import defaultdict

import torch


@torch.no_grad()
def r2_score(x, x_hat, eps=1e-6):
    """
    R^2 as reported in the LG-SAE paper (same formula as ``overcomplete``)::

        1 - mean((x - x_hat)^2) / mean((x - mean(x))^2)

    where ``mean(x)`` is a single scalar over *all* entries of ``x``, not a
    per-dimension mean. When input dimensions have different means this is
    larger than the per-dimension (fraction-of-variance-explained) R^2.

    ``x`` and ``x_hat`` must be 2D with identical shapes.
    """
    if x.shape != x_hat.shape or x.dim() != 2:
        raise ValueError(f"r2_score expects two 2D tensors of equal shape, got {tuple(x.shape)} "
                         f"and {tuple(x_hat.shape)}")
    ss_res = (x - x_hat).square().mean()
    ss_tot = (x - x.mean()).square().mean()
    return (1.0 - ss_res / (ss_tot + eps)).item()


@torch.no_grad()
def mean_l0(z):
    """Average number of active (strictly positive) latents per row of ``z``."""
    return (z > 0).sum(dim=-1).float().mean().item()


@torch.no_grad()
def nonzero_fraction(t):
    """Fraction of strictly positive entries of a non-negative tensor."""
    return (t > 0).float().mean().item()


class DeadLatentTracker:
    """Fraction of latents that were never active on any row seen since construction."""

    def __init__(self, n_latents, device):
        self.alive = torch.zeros(n_latents, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, z):
        self.alive |= (z > 0).any(dim=0)

    def dead_ratio(self):
        return 1.0 - self.alive.float().mean().item()


class RunningMeans:
    """Accumulates scalar metrics and returns their means."""

    def __init__(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, **metrics):
        for name, value in metrics.items():
            self._sums[name] += float(value)
            self._counts[name] += 1

    def means(self):
        return {name: self._sums[name] / self._counts[name] for name in self._sums}


def format_metrics(metrics):
    """Compact one-line rendering; ratios are shown as percentages."""
    parts = []
    for name, value in metrics.items():
        if name.endswith("ratio") or name.endswith("density"):
            parts.append(f"{name}={100 * value:.2f}%")
        elif name.startswith("L0") or name.endswith("per_row"):
            parts.append(f"{name}={value:.2f}")
        elif value == 0 or abs(value) >= 1e-3:
            parts.append(f"{name}={value:.4f}")
        else:
            parts.append(f"{name}={value:.3e}")
    return "  ".join(parts)
