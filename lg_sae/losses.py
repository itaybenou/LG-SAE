"""Reconstruction losses for TopK / BatchTopK SAEs."""

import torch


def aux_k_mse(residual, pre_codes, codes, dictionary):
    """
    AuxK term (Gao et al., 2024): reconstruct the residual ``x - x_hat`` from
    the strongest *inactive* latents (the top half of ``ReLU(pre_codes)``
    among the latents that were zeroed by the sparsity constraint). This
    keeps gradient flowing into latents that would otherwise die.
    """
    inactive = torch.relu(pre_codes) - codes
    top = torch.topk(inactive, k=inactive.shape[-1] // 2, dim=-1)
    aux_codes = torch.zeros_like(codes).scatter(-1, top.indices, top.values)
    return (residual - aux_codes @ dictionary).square().mean()


def sae_loss(x, x_hat, pre_codes, codes, dictionary, aux_penalty):
    """
    ``MSE(x, x_hat) + aux_penalty * AuxK``.

    Returns ``(total, mse, aux_mse)`` so each part can be logged.
    """
    residual = x - x_hat
    mse = residual.square().mean()
    aux_mse = aux_k_mse(residual, pre_codes, codes, dictionary)
    return mse + aux_penalty * aux_mse, mse, aux_mse
