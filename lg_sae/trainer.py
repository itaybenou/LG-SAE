"""
Training loops for the four LG-SAE stages.

- ``train_sae_component``: stage 1 (local SAE on patch tokens) and stage 2
  (global SAE on image embeddings), each trained in isolation.
- ``train_bridge``: stage 3 and stage 4b; only ``C`` and ``link_bias`` train.
- ``finetune_local``: stage 4a; the local SAE is fine-tuned so its pooled
  codes also serve the (frozen) bridge and global SAE.

Every loop returns a history dict with per-epoch means (``"epoch"``) and the
running means printed every ``log_interval`` steps (``"interval"``).
"""

import time

import torch

from .losses import sae_loss
from .metrics import (DeadLatentTracker, RunningMeans, format_metrics, mean_l0,
                      nonzero_fraction, r2_score)


def _optimize(optimizer, loss, params, clip_grad):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(params, clip_grad)
    optimizer.step()


def _run_epochs(name, epochs, dataset, make_epoch, log_interval):
    """
    Generic epoch loop. ``make_epoch()`` is called at the start of each epoch
    and returns ``(step_fn, summary_fn)``: ``step_fn(batch)`` trains on one
    batch and returns a dict of scalar metrics (averaged over the epoch);
    ``summary_fn()`` returns epoch-level metrics (e.g. dead-latent ratio).
    """
    history = {"epoch": [], "interval": []}
    for epoch in range(1, epochs + 1):
        step_fn, summary_fn = make_epoch()
        meter = RunningMeans()
        start = time.time()
        n_steps = 0
        for batch in dataset:
            n_steps += 1
            meter.update(**step_fn(batch))
            if log_interval > 0 and n_steps % log_interval == 0:
                stats = {**meter.means(), **summary_fn()}
                history["interval"].append({"epoch": epoch, "step": n_steps, **stats})
                print(f"    [{name}] epoch {epoch}/{epochs} step {n_steps} | {format_metrics(stats)}",
                      flush=True)
        if n_steps == 0:
            raise RuntimeError(f"[{name}] no batches produced; the batch size is larger than every shard")
        stats = {**meter.means(), **summary_fn()}
        elapsed = time.time() - start
        history["epoch"].append({"epoch": epoch, "steps": n_steps, "time_s": elapsed, **stats})
        print(f"  [{name}] epoch {epoch}/{epochs} ({n_steps} steps, {elapsed:.1f}s) | "
              f"{format_metrics(stats)}", flush=True)
    return history


def train_sae_component(model, component, dataset, *, epochs, lr, aux_penalty, clip_grad,
                        device, log_interval, name):
    """
    Train ``model.local_sae`` (``component="local"``, batches of patch tokens)
    or ``model.global_sae`` (``component="global"``, batches of image
    embeddings) alone with ``MSE + aux_penalty * AuxK``.
    """
    if component not in ("local", "global"):
        raise ValueError(f"component must be 'local' or 'global', got {component!r}")
    is_local = component == "local"
    model.set_trainable(local_encoder=is_local, local_dictionary=is_local,
                        global_sae=not is_local, bridge=False)
    sae = model.local_sae if is_local else model.global_sae
    sae.train()
    params = [p for p in sae.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    def make_epoch():
        tracker = DeadLatentTracker(sae.n_latents, device)

        def step(x):
            x = x.to(device, non_blocking=True)
            pre_codes, z, x_hat = sae(x)
            loss, mse, aux_mse = sae_loss(x, x_hat, pre_codes, z, sae.get_dictionary(), aux_penalty)
            _optimize(optimizer, loss, params, clip_grad)
            tracker.update(z)
            return {"loss": loss.item(), "mse": mse.item(), "aux_mse": aux_mse.item(),
                    "R2": r2_score(x, x_hat), "L0": mean_l0(z)}

        return step, lambda: {"dead_ratio": tracker.dead_ratio()}

    return _run_epochs(name, epochs, dataset, make_epoch, log_interval)


@torch.no_grad()
def _bridge_summary(model):
    C = model.get_C()
    return {"C_density": nonzero_fraction(C), "C_nnz_per_row": (C > 0).sum(dim=1).float().mean().item()}


def train_bridge(model, dataset, *, epochs, lr, w_rec_bridge, l1_penalty, clip_grad,
                 device, log_interval, name):
    """
    Train the bridge (``C``, ``link_bias``) with both SAEs frozen::

        loss = MSE(z_global, z_pred) + w_rec_bridge * MSE(x_global, z_pred @ D_global)
               + l1_penalty * mean(|C|)
    """
    model.set_trainable(local_encoder=False, local_dictionary=False, global_sae=False, bridge=True)
    model.local_sae.eval()
    model.global_sae.eval()
    params = model.bridge_parameters()
    optimizer = torch.optim.Adam(params, lr=lr)

    def make_epoch():
        def step(batch):
            x_global, patches = (t.to(device, non_blocking=True) for t in batch)
            with torch.no_grad():
                _, z_global = model.encode_global(x_global)
                _, z_local = model.encode_local(patches)
                z_pooled = model.pool(z_local, x_global.shape[0])
            C = model.get_C()
            z_pred = model.predict_global_codes(z_pooled, C)
            x_pred = model.decode_global(z_pred)

            L_link = (z_global - z_pred).square().mean()
            L_rec_bridge = (x_global - x_pred).square().mean()
            L_sparse_C = C.abs().mean()
            loss = L_link + w_rec_bridge * L_rec_bridge + l1_penalty * L_sparse_C
            _optimize(optimizer, loss, params, clip_grad)
            return {"loss": loss.item(), "L_link": L_link.item(), "L_rec_bridge": L_rec_bridge.item(),
                    "L_sparse_C": L_sparse_C.item(), "R2_bridge": r2_score(x_global, x_pred),
                    "R2_link": r2_score(z_global, z_pred), "L0_pred": mean_l0(z_pred)}

        return step, lambda: _bridge_summary(model)

    return _run_epochs(name, epochs, dataset, make_epoch, log_interval)


def finetune_local(model, dataset, *, epochs, lr, aux_penalty, w_rec_bridge, w_link,
                   freeze_local_dictionary, clip_grad, device, log_interval, name):
    """
    Fine-tune the local SAE with the global SAE and the bridge frozen::

        loss = MSE(patches, patches_hat) + aux_penalty * AuxK
               + w_rec_bridge * MSE(x_global, z_pred @ D_global)
               + w_link * MSE(z_global, z_pred)

    where ``z_pred = ReLU(pool(z_local) @ C^T + b)``, so the bridge terms
    backpropagate into the local encoder. With ``freeze_local_dictionary``
    only the local encoder is updated.
    """
    model.set_trainable(local_encoder=True, local_dictionary=not freeze_local_dictionary,
                        global_sae=False, bridge=False)
    model.global_sae.eval()
    model.local_sae.train()
    if freeze_local_dictionary:
        model.local_sae.dictionary.eval()
    params = [p for p in model.local_sae.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    def make_epoch():
        tracker = DeadLatentTracker(model.n_local, device)

        def step(batch):
            x_global, patches = (t.to(device, non_blocking=True) for t in batch)
            patches_flat = patches.reshape(-1, patches.shape[-1])
            pre_codes, z_local = model.local_sae.encode(patches_flat)
            patches_hat = model.decode_local(z_local)
            L_local_rec, mse, aux_mse = sae_loss(patches_flat, patches_hat, pre_codes, z_local,
                                                 model.local_sae.get_dictionary(), aux_penalty)

            with torch.no_grad():
                _, z_global = model.encode_global(x_global)
            z_pred = model.predict_global_codes(model.pool(z_local, x_global.shape[0]))
            x_pred = model.decode_global(z_pred)
            L_rec_bridge = (x_global - x_pred).square().mean()
            L_link = (z_global - z_pred).square().mean()

            loss = L_local_rec + w_rec_bridge * L_rec_bridge + w_link * L_link
            _optimize(optimizer, loss, params, clip_grad)
            tracker.update(z_local)
            return {"loss": loss.item(), "L_local_rec": L_local_rec.item(), "mse": mse.item(),
                    "aux_mse": aux_mse.item(), "L_rec_bridge": L_rec_bridge.item(),
                    "L_link": L_link.item(), "R2": r2_score(patches_flat, patches_hat),
                    "L0": mean_l0(z_local), "R2_bridge": r2_score(x_global, x_pred)}

        return step, lambda: {"dead_ratio": tracker.dead_ratio()}

    return _run_epochs(name, epochs, dataset, make_epoch, log_interval)
