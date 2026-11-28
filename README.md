# **Local–Global Sparse Autoencoders for Multiscale Interpretability in Vision Models [NeurIPS'26]**



![Open the Concept Explorer](https://img.shields.io/badge/Open%20the%20Concept%20Explorer-1f6feb?style=for-the-badge&labelColor=0f172a)

## Installation

1. Clone the repository.
  ```bash
   git clone https://github.com/itaybenou/LG-SAE.git
   cd LG-SAE
  ```
2. Create an environment with Python 3.10 and PyTorch 2.1 or newer, with a matching torchvision build.
  ```bash
   conda create -n lg-sae python=3.10
   conda activate lg-sae
  ```
3. Install the remaining dependencies.
  ```bash
   pip install -r requirements.txt
  ```
4. Download the ImageNet training and validation sets and arrange each split as a torchvision `ImageFolder`: one subdirectory per synset, with the images inside it.
  ```text
   data/imagenet/
     train/
       n01440764/
         n01440764_18.JPEG
         ...
       n01443537/
         ...
     val/
       n01440764/
         ILSVRC2012_val_00000293.JPEG
         ...
       n01443537/
         ...
  ```
   `save_shards.py` and the naming scripts both read an `ImageFolder` root, so `imagenet/train` and `imagenet/val` are the directories passed as `--image_dir`.
5. **For the PartImageNet evaluation:** Download [PartImageNet](https://github.com/TACJu/PartImageNet) and arrange the test split as a flat image directory, a flat directory of label maps, and one ontology file. The image id is the file name without its extension, and the label map for that image must be `<image_id>.png`.
  ```text
   data/PartImageNet/
     images/test/
       n01440764_123.JPEG
       ...
     annotations/test/
       n01440764_123.png
       ...
     gt_ontology.json
  ```

## Usage

Paths below are placeholders. Config files hold the model and training hyperparameters, and every path is passed on the command line.

### Training LG-SAE

Training reads pre-extracted features, so run `save_shards.py` first. It writes aligned shards:

```text
<output_dir>/global/shard_00000.pt         (n, d_global)
<output_dir>/patch_tokens/shard_00000.pt   (n, T, d_local)
<output_dir>/idxs/shard_00000.pt
<output_dir>/labels/shard_00000.pt
<output_dir>/manifest.json
```

For example, for CLIP ViT-B/16 on the ImageNet training set:

```bash
python save_shards.py --config configs/shards/clip_vitb16.yaml \
    --image_dir /data/imagenet/train \
    --output_dir /data/features/clip_vitb16/train
```

For DINOv2 ViT-B/14, use `configs/shards/dinov2_vitb14.yaml`. On a new machine, set `hub_repo` in that file to `facebookresearch/dinov2` so `torch.hub` can download the backbone.

**Training:** The example below trains a BatchTopK configuration on top of CLIP backbone, as configured by `configs/train/clip_batchtopk.yaml`. 

```bash
python train_lg_sae.py --config configs/train/clip_batchtopk.yaml \
    --global_dir /data/features/clip_vitb16/train/global \
    --patch_dir /data/features/clip_vitb16/train/patch_tokens \
    --save_dir runs/clip_batchtopk \
    --device cuda:0
```

### Labeling LG-SAE features

Naming compares SAE codes with CLIP on a probing set. Extract that set with `save_shards.py` and `--shuffle false`, so row `i` is the `i`-th image of the `ImageFolder` in sorted order. The commands below use the ImageNet validation set.

Extract the validation features once:

```bash
python save_shards.py --config configs/shards/clip_vitb16.yaml \
    --image_dir /data/imagenet/val \
    --output_dir /data/features/clip_vitb16/val \
    --shuffle false
```

Label global features:

```bash
python describe_global_neurons.py --config configs/describe/global_clip.yaml \
    --activations none \
    --sae_checkpoint runs/clip_batchtopk/final.pt \
    --model_config runs/clip_batchtopk/config.json \
    --global_features /data/features/clip_vitb16/val/global \
    --sae_batch_size 64 \
    --image_dir /data/imagenet/val \
    --sample_stride 1 \
    --concept_set concept_sets/10k.txt \
    --output_dir runs/clip_batchtopk/describe_global
```

Label local features:  
`--dinoiser_ckpt` is the released [CLIP-DINOiser](https://github.com/wysoczanska/clip_dinoiser) file `checkpoints/last.pt`.

```bash
python describe_local_neurons.py --config configs/describe/local_dinoiser.yaml \
    --activations none \
    --sae_checkpoint runs/clip_batchtopk/final.pt \
    --model_config runs/clip_batchtopk/config.json \
    --patch_features /data/features/clip_vitb16/val/patch_tokens \
    --sae_batch_size 64 \
    --image_dir /data/imagenet/val \
    --sample_stride 10 \
    --concept_set concept_sets/20k.txt \
    --dinoiser_ckpt /data/clip_dinoiser/last.pt \
    --output_dir runs/clip_batchtopk/describe_local
```

For a DINOv2 SAE, extract the ImageNet validation features with `configs/shards/dinov2_vitb14.yaml` and `--shuffle false`, after the same `hub_repo` change as in training. Point `describe_global_neurons.py` at that run's `val/global` directory. Point `--patch_features` at the DINOv2 `val/patch_tokens` directory and pass `--image_size 256`, so the CLIP-DINOiser grid is 16×16.

### Compositional structure recovery on Part-ImageNet

Extract test features with the same backbone and token layer used to train the SAE. 

```bash
python extract_partimagenet_features.py --config configs/partimagenet/clip_vitb16.yaml \
    --image_dir /data/PartImageNet/images/test \
    --output_dir /data/partimagenet/clip_vitb16
```

Use `configs/partimagenet/dinov2_vitb14.yaml`for DINOv2 SAE. Apply the same `hub_repo` change as in training.

**Evaluation:**

```bash
python eval_partimagenet_composition.py --config configs/eval/partimagenet_composition.yaml \
    --checkpoint runs/clip_batchtopk/final.pt \
    --model_config runs/clip_batchtopk/config.json \
    --patch_tokens /data/partimagenet/clip_vitb16/patch_tokens.pt \
    --global_features /data/partimagenet/clip_vitb16/global.pt \
    --image_ids /data/partimagenet/clip_vitb16/image_ids.json \
    --labelmap_dir /data/PartImageNet/annotations/test \
    --ontology /data/PartImageNet/gt_ontology.json \
    --freq_patch_file /data/features/clip_vitb16/train/patch_tokens/shard_00000.pt \
    --output_dir PartImageNet_composition_eval/clip_batchtopk
```

