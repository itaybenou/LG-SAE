"""
Sparse autoencoders for LG-SAE.

- ``TopKSAE``: keeps the ``k`` largest activations of every sample.
- ``BatchTopKSAE``: keeps the ``k * batch_size`` largest activations of the
  whole batch (Bussmann et al., 2024), so ``k`` is the *average* number of
  active latents per sample. A running (EMA) threshold is used at eval time.
- ``LocalGlobalSAE``: a local SAE on patch tokens, a global SAE on image-level
  embeddings, and a non-negative bridge ``C`` that predicts global codes from
  pooled local codes: ``z_global_pred = ReLU(pool(z_local) @ C^T + b)``.

State-dict keys are compatible with the ``overcomplete`` CoCo-SAE v2
checkpoints (see ``load_components``).
"""

import torch
import torch.nn.functional as F
from torch import nn

SAE_TYPES = ("topk", "batchtopk")
ENCODER_TYPES = ("linear", "mlp_ln_1")
POOLINGS = ("mean", "max", "softmax", "active_mean")
COMPONENTS = ("local", "global", "bridge")


def _l2_normalize_rows(w):
    return w / (w.norm(p=2, dim=1, keepdim=True) + 1e-8)


class DictionaryLayer(nn.Module):
    """
    Dictionary ``D`` of shape ``(n_latents, dim)`` with unit-norm atoms.

    In training mode the atoms are projected onto the unit sphere in place
    every time the dictionary is read (projected gradient descent), so the
    optimizer always steps from normalized weights. In eval mode the
    normalized dictionary is returned without touching the weights.
    """

    def __init__(self, dim, n_latents, device="cpu"):
        super().__init__()
        self._weights = nn.Parameter(torch.empty(n_latents, dim, device=device))
        nn.init.xavier_uniform_(self._weights)

    def get_dictionary(self):
        if self.training:
            with torch.no_grad():
                self._weights.data = _l2_normalize_rows(self._weights)
            return self._weights
        return _l2_normalize_rows(self._weights)

    def forward(self, z):
        return z @ self.get_dictionary()


class Encoder(nn.Module):
    """
    Affine encoder returning ``(pre_codes, ReLU(pre_codes))``.

    ``linear``:   pre_codes = W x + b
    ``mlp_ln_1``: pre_codes = LayerNorm(W x + b)
    """

    def __init__(self, input_dim, n_latents, encoder_type="linear", device="cpu"):
        super().__init__()
        if encoder_type not in ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {ENCODER_TYPES}, got {encoder_type!r}")
        norm = nn.LayerNorm(n_latents, device=device) if encoder_type == "mlp_ln_1" else nn.Identity()
        # Attribute name matches overcomplete's MLPEncoder so checkpoints load as-is.
        self.final_block = nn.Sequential(nn.Linear(input_dim, n_latents, device=device), norm)

    def forward(self, x):
        pre_codes = self.final_block(x)
        return pre_codes, F.relu(pre_codes)


class TopKSAE(nn.Module):
    """
    Per-sample TopK SAE: ``z = TopK_k(ReLU(enc(x)))``, ``x_hat = z @ D``.

    Parameters
    ----------
    input_dim : int
        Dimension of the input activations.
    n_latents : int
        Number of dictionary atoms (concepts).
    k : int
        Number of active latents per sample.
    encoder_type : str
        One of ``ENCODER_TYPES``.
    device : str
    """

    def __init__(self, input_dim, n_latents, k, encoder_type="linear", device="cpu"):
        super().__init__()
        if not 0 < k <= n_latents:
            raise ValueError(f"k must be in [1, {n_latents}], got {k}")
        self.input_dim = input_dim
        self.n_latents = n_latents
        self.k = k
        self.encoder = Encoder(input_dim, n_latents, encoder_type, device=device)
        self.dictionary = DictionaryLayer(input_dim, n_latents, device=device)

    def sparsify(self, codes):
        top = torch.topk(codes, self.k, dim=-1)
        return torch.zeros_like(codes).scatter(-1, top.indices, top.values)

    def encode(self, x):
        """``x``: ``(N, input_dim)`` -> ``(pre_codes, z)``, both ``(N, n_latents)``."""
        pre_codes, codes = self.encoder(x)
        return pre_codes, self.sparsify(codes)

    def decode(self, z):
        return self.dictionary(z)

    def get_dictionary(self):
        return self.dictionary.get_dictionary()

    def forward(self, x):
        pre_codes, z = self.encode(x)
        return pre_codes, z, self.decode(z)


class BatchTopKSAE(TopKSAE):
    """
    BatchTopK SAE: in training, keeps the ``k * N`` largest activations of a
    batch of ``N`` rows (so ``k`` is the mean number of active latents per row,
    independent of the batch size). The ``k * N``-th value is tracked with an
    EMA (``running_threshold``) and used as a fixed threshold in eval mode.
    """

    def __init__(self, input_dim, n_latents, k, threshold_momentum=0.9,
                 encoder_type="linear", device="cpu"):
        super().__init__(input_dim, n_latents, k, encoder_type, device)
        if not 0.0 <= threshold_momentum < 1.0:
            raise ValueError(f"threshold_momentum must be in [0, 1), got {threshold_momentum}")
        self.threshold_momentum = threshold_momentum
        self.register_buffer("running_threshold", torch.tensor(float("nan"), device=device))

    def sparsify(self, codes):
        if self.training:
            n_keep = min(self.k * codes.shape[0], codes.numel())
            threshold = torch.topk(codes.flatten(), n_keep).values[-1].detach()
            if torch.isnan(self.running_threshold):
                self.running_threshold.copy_(threshold)
            else:
                self.running_threshold.lerp_(threshold, 1.0 - self.threshold_momentum)
        else:
            if torch.isnan(self.running_threshold):
                raise RuntimeError("BatchTopKSAE running_threshold is not initialized; "
                                   "train the SAE or load a checkpoint first.")
            threshold = self.running_threshold
        return codes * (codes >= threshold).to(codes.dtype)


def build_sae(sae_type, input_dim, n_latents, k, encoder_type, threshold_momentum=None, device="cpu"):
    """Instantiate a ``TopKSAE`` or ``BatchTopKSAE`` from config values."""
    if sae_type == "topk":
        return TopKSAE(input_dim, n_latents, k, encoder_type, device=device)
    if sae_type == "batchtopk":
        if threshold_momentum is None:
            raise ValueError("threshold_momentum is required for sae_type='batchtopk'")
        return BatchTopKSAE(input_dim, n_latents, k, threshold_momentum, encoder_type, device=device)
    raise ValueError(f"sae_type must be one of {SAE_TYPES}, got {sae_type!r}")


class LocalGlobalSAE(nn.Module):
    """
    Local SAE (patch tokens) + global SAE (image embeddings) + bridge.

    The bridge predicts global codes from pooled local codes::

        z_global_pred = ReLU(pool(z_local) @ C^T + link_bias),   C = ReLU(_C) >= 0

    Parameters
    ----------
    local_sae, global_sae : TopKSAE or BatchTopKSAE
    pooling : str
        How patch codes ``(B, T, n_local)`` are pooled to ``(B, n_local)``;
        one of ``POOLINGS``.
    c_top_k : int or None
        If set, each row of ``C`` keeps only its ``c_top_k`` largest entries in
        the forward pass (straight-through gradient to all entries).
    """

    def __init__(self, local_sae, global_sae, pooling="mean", c_top_k=None):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        if c_top_k is not None and not 0 < c_top_k <= local_sae.n_latents:
            raise ValueError(f"c_top_k must be in [1, {local_sae.n_latents}] or None, got {c_top_k}")
        self.local_sae = local_sae
        self.global_sae = global_sae
        self.pooling = pooling
        self.c_top_k = c_top_k

        device = local_sae.dictionary._weights.device
        self._C = nn.Parameter(torch.rand(global_sae.n_latents, local_sae.n_latents, device=device) * 0.01)
        self.link_bias = nn.Parameter(torch.zeros(global_sae.n_latents, device=device))

    @property
    def n_local(self):
        return self.local_sae.n_latents

    @property
    def n_global(self):
        return self.global_sae.n_latents

    # ------------------------------------------------------------------ streams

    def encode_local(self, patches):
        """``patches``: ``(B, T, d_local)`` -> ``(pre_codes, z)``, both ``(B*T, n_local)``."""
        return self.local_sae.encode(patches.reshape(-1, patches.shape[-1]))

    def decode_local(self, z_local):
        return self.local_sae.decode(z_local)

    def encode_global(self, x):
        """``x``: ``(B, d_global)`` -> ``(pre_codes, z)``, both ``(B, n_global)``."""
        return self.global_sae.encode(x)

    def decode_global(self, z_global):
        return self.global_sae.decode(z_global)

    # ------------------------------------------------------------------ bridge

    def pool(self, z_local, batch_size):
        """Pool flat patch codes ``(B*T, n_local)`` into image codes ``(B, n_local)``."""
        z = z_local.reshape(batch_size, -1, z_local.shape[-1])
        if self.pooling == "mean":
            return z.mean(dim=1)
        if self.pooling == "max":
            return z.max(dim=1).values
        if self.pooling == "softmax":
            return (F.softmax(z, dim=1) * z).sum(dim=1)
        active = (z > 0).to(z.dtype)
        return (z * active).sum(dim=1) / active.sum(dim=1).clamp(min=1.0)

    def get_C(self):
        """Non-negative composition matrix ``(n_global, n_local)``."""
        C = F.relu(self._C)
        if self.c_top_k is not None:
            top = torch.topk(C, self.c_top_k, dim=-1)
            C_hard = torch.zeros_like(C).scatter(-1, top.indices, top.values)
            C = C + (C_hard - C).detach()
        return C

    def predict_global_codes(self, z_local_pooled, C=None):
        """``(B, n_local)`` pooled local codes -> ``(B, n_global)`` predicted global codes."""
        if C is None:
            C = self.get_C()
        return F.relu(z_local_pooled @ C.T + self.link_bias)

    # ------------------------------------------------------------------ training helpers

    def bridge_parameters(self):
        return [self._C, self.link_bias]

    def set_trainable(self, *, local_encoder, local_dictionary, global_sae, bridge):
        """Set ``requires_grad`` per component (keyword-only to keep call sites explicit)."""
        for p in self.local_sae.encoder.parameters():
            p.requires_grad_(local_encoder)
        for p in self.local_sae.dictionary.parameters():
            p.requires_grad_(local_dictionary)
        for p in self.global_sae.parameters():
            p.requires_grad_(global_sae)
        for p in self.bridge_parameters():
            p.requires_grad_(bridge)


# ---------------------------------------------------------------------- checkpoints

_COMPONENT_PREFIXES = {
    "local": ("local_sae.",),
    "global": ("global_sae.",),
    "bridge": ("_C", "link_bias"),
}

# overcomplete's per-sample TopK model (CompositionalConceptSAEv2) stored the
# encoders/dictionaries at the top level instead of inside local_sae/global_sae.
_LEGACY_RENAMES = {
    "local_encoder.": "local_sae.encoder.",
    "local_dictionary.": "local_sae.dictionary.",
    "global_encoder.": "global_sae.encoder.",
    "global_dictionary.": "global_sae.dictionary.",
}


def _normalize_state_dict(state_dict):
    """Rename legacy keys and drop overcomplete's (always-zero) dictionary multipliers."""
    out = {}
    for key, value in state_dict.items():
        for old, new in _LEGACY_RENAMES.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        if key.endswith("dictionary.multiplier"):
            if value.abs().max().item() != 0.0:
                raise ValueError(f"Unsupported non-zero dictionary multiplier in checkpoint: {key}")
            continue
        out[key] = value
    return out


def load_components(model, checkpoint_path, components, device="cpu"):
    """
    Load the selected components (``"local"``, ``"global"``, ``"bridge"``) of
    ``model`` from a checkpoint written by this repo or by overcomplete.

    Raises if any parameter/buffer of a selected component is missing.
    Returns the number of loaded tensors.
    """
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown components {sorted(unknown)}; expected a subset of {COMPONENTS}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = _normalize_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    prefixes = tuple(p for c in components for p in _COMPONENT_PREFIXES[c])
    selected = {k: v for k, v in state_dict.items() if k.startswith(prefixes)}
    expected = {k for k in model.state_dict() if k.startswith(prefixes)}
    missing = sorted(expected - selected.keys())
    if missing:
        raise KeyError(f"{checkpoint_path} is missing {len(missing)} tensors for "
                       f"{list(components)}, e.g. {missing[:5]}")
    unexpected = sorted(selected.keys() - expected)
    if unexpected:
        raise KeyError(f"{checkpoint_path} has tensors not in the model: {unexpected[:5]}")

    model.load_state_dict(selected, strict=False)
    return len(selected)
