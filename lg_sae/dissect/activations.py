"""
SAE activations of a probing set, from pre-extracted backbone features.

Features are one ``.pt``/``.pth`` file or a directory of shards (see
``lg_sae.data``), in the same order as ``ImageFolder(image_dir)``:

- local:  patch tokens ``(N, T, d_local)``
- global: image embeddings ``(N, d_global)``

``sample_stride`` selects the same rows as the probing ``ImageFolder`` subset.
"""

import os

import torch
import yaml
from torchvision import datasets
from tqdm import tqdm

from lg_sae.config import ConfigError
from lg_sae.data import list_shards
from lg_sae.partimagenet import patch_grid_size
from lg_sae.sae import LocalGlobalSAE, build_sae, load_components

from .common import load_activations

MODEL_KEYS = ("sae_type", "n_local", "n_global", "k_local", "k_global", "local_encoder",
              "global_encoder", "threshold_momentum", "pooling", "c_top_k")


def _load_tensor(path):
    """Load a feature tensor, memory-mapped when the file supports it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, ValueError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=True)


def _require(cfg, names):
    missing = [name for name in names if getattr(cfg, name) is None]
    if missing:
        raise ConfigError(f"activations is null, so these options must be set: {missing}")


def _inspect(path, kind):
    """``(files, n_rows, feature_dim, n_tokens)``. ``n_tokens`` is None for global features."""
    files = list_shards(path)
    n_rows = 0
    feature_dim = n_tokens = None
    for shard in files:
        tensor = _load_tensor(shard)
        if kind == "local":
            if tensor.dim() != 3:
                raise ValueError(f"Expected patch tokens (N, T, d) in {shard}, got {tuple(tensor.shape)}")
            tokens, dim = tensor.shape[1], tensor.shape[2]
            if n_tokens is None:
                n_tokens, feature_dim = tokens, dim
            elif (tokens, dim) != (n_tokens, feature_dim):
                raise ValueError(f"{shard} has shape {tuple(tensor.shape)}, expected "
                                 f"(N, {n_tokens}, {feature_dim})")
        else:
            if tensor.dim() != 2:
                raise ValueError(f"Expected image embeddings (N, d) in {shard}, got {tuple(tensor.shape)}")
            dim = tensor.shape[1]
            if feature_dim is None:
                feature_dim = dim
            elif dim != feature_dim:
                raise ValueError(f"{shard} has shape {tuple(tensor.shape)}, expected (N, {feature_dim})")
        n_rows += tensor.shape[0]
    if n_rows == 0:
        raise ValueError(f"No rows in {path}")
    return files, n_rows, feature_dim, n_tokens


def _load_model(checkpoint, model_config, feature_dim, kind, device):
    with open(model_config) as f:
        mc = yaml.safe_load(f)
    missing = [key for key in MODEL_KEYS if key not in mc]
    if missing:
        raise KeyError(f"{model_config} is missing model keys {missing}")
    # Only one stream is encoded; the other SAE is a placeholder and is not loaded.
    local_dim = feature_dim if kind == "local" else 1
    global_dim = feature_dim if kind == "global" else 1
    model = LocalGlobalSAE(
        local_sae=build_sae(mc["sae_type"], local_dim, mc["n_local"], mc["k_local"],
                            mc["local_encoder"], mc["threshold_momentum"], device),
        global_sae=build_sae(mc["sae_type"], global_dim, mc["n_global"], mc["k_global"],
                             mc["global_encoder"], mc["threshold_momentum"], device),
        pooling=mc["pooling"], c_top_k=mc["c_top_k"])
    load_components(model, checkpoint, ("local",) if kind == "local" else ("global",), device)
    return model.eval()


@torch.no_grad()
def _encode(model, files, kind, stride, batch_size, device, n_out, n_tokens):
    if kind == "local":
        grid = patch_grid_size(n_tokens)
        out = torch.empty((n_out, model.n_local, grid, grid), dtype=torch.float16)
    else:
        grid = None
        out = torch.empty((n_out, model.n_global), dtype=torch.float32)

    write = 0
    offset = 0
    for shard in tqdm(files, desc=f"z_{kind}"):
        tensor = _load_tensor(shard)
        start = (stride - offset % stride) % stride
        rows = tensor[start::stride]
        for batch_start in range(0, rows.shape[0], batch_size):
            batch = rows[batch_start:batch_start + batch_size].float().to(device)
            if kind == "local":
                _, z = model.encode_local(batch)
                b = batch.shape[0]
                z = z.reshape(b, grid, grid, model.n_local).permute(0, 3, 1, 2).half()
            else:
                _, z = model.encode_global(batch)
                z = z.float()
            out[write:write + z.shape[0]] = z.cpu()
            write += z.shape[0]
        offset += tensor.shape[0]
        del tensor
    if write != n_out:
        raise RuntimeError(f"Encoded {write} rows, expected {n_out}")
    return out


def resolve_activations(cfg, kind):
    """
    Load ``cfg.activations``, or compute them when it is null.

    Computed codes are written to ``output_dir/z_local.pt`` or ``output_dir/z_global.pt``
    and ``cfg.activations`` is set to that path.
    """
    if kind not in ("local", "global"):
        raise ValueError(f"kind must be 'local' or 'global', got {kind!r}")
    ndim = 4 if kind == "local" else 2
    if cfg.activations is not None:
        return load_activations(cfg.activations, ndim)

    feature_name = "patch_features" if kind == "local" else "global_features"
    _require(cfg, ["sae_checkpoint", "model_config", "sae_batch_size", feature_name])
    if cfg.sample_stride < 1:
        raise ValueError(f"sample_stride must be >= 1, got {cfg.sample_stride}")
    if cfg.sae_batch_size < 1:
        raise ValueError(f"sae_batch_size must be >= 1, got {cfg.sae_batch_size}")

    feature_path = getattr(cfg, feature_name)
    files, n_rows, feature_dim, n_tokens = _inspect(feature_path, kind)
    n_images = len(datasets.ImageFolder(cfg.image_dir))
    if n_rows != n_images:
        raise ValueError(f"{feature_path} has {n_rows} rows but {cfg.image_dir} has {n_images} images. "
                         f"Features must follow sorted ImageFolder order so sample_stride={cfg.sample_stride} "
                         "selects the same rows.")

    n_out = len(range(0, n_rows, cfg.sample_stride))
    print(f"computing z_{kind} from {feature_path} ({n_rows} rows, sample_stride={cfg.sample_stride} "
          f"-> {n_out})")
    model = _load_model(cfg.sae_checkpoint, cfg.model_config, feature_dim, kind, cfg.device)
    acts = _encode(model, files, kind, cfg.sample_stride, cfg.sae_batch_size, cfg.device, n_out, n_tokens)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    path = os.path.abspath(os.path.join(cfg.output_dir, f"z_{kind}.pt"))
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(acts, path)
    print(f"Saved z_{kind} {tuple(acts.shape)} to {path}")
    cfg.activations = path
    return acts
