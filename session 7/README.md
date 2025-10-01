## CIFAR-10 Experiments and Modular PyTorch Pipeline

This repository documents an iterative journey of building and improving a CNN for CIFAR-10, along with a clean, modular PyTorch training pipeline extracted from the final notebook.

### Highlights
- Optimized `ciferNet` CNN targeting receptive field ≈ 44 and ≈ 200k parameters
- Depthwise separable convolutions, controlled dilation, and efficient downsampling
- Albumentations-based data augmentation to reduce overfitting
- Modular Python package with `models/`, `data/`, `engine/`, and `main.py`

---

## Experiment Timeline (Notebooks)

### 1) `colab_files/cifar10.ipynb`
- Verified dataset integrity, image sizes, and channel details.

### 2) `colab_files/cifar10_transform_values.ipynb`
- Explored and validated augmentation/normalization values for CIFAR-10.

### 3) `colab_files/cifar10_v1.ipynb`
- Built a baseline CNN (~380k params) and trained for 8 epochs.
- Purpose: establish a working skeleton to iterate and improve upon.

### 4) `colab_files/cifar10_v2.ipynb`
- Reduced parameters by introducing a 1×1 bottleneck before ConvBlock4.
- Goal: shrink ConvBlock4 while retaining capacity and receptive field structure.

### 5) `colab_files/cifar10_v3.ipynb`
- Switched to depthwise + pointwise (separable) convolutions.
- Params reduced to ~171k; trained 8 epochs; reached ~77% accuracy.

### 6) `colab_files/cifar10_v4.ipynb`
- Observed overfitting; added dropout across layers (≈170k params).
- No meaningful accuracy improvement; moved focus to augmentation.

### 7) `colab_files/cifar10_v5.ipynb`
- Introduced Albumentations (HorizontalFlip, ShiftScaleRotate, CoarseDropout).
- Clear regularization gains; at 35 epochs achieved ~84% accuracy.

### 8) `colab_files/cifar10_v6.ipynb`
- Tuned receptive field growth and downsampling:
  - Dilation: 2 (controlled RF expansion)
  - Kernel size: reduced (e.g., 5×5 in that iteration) to avoid excessive RF
  - Stride: 2 for gradual downsampling
  - Padding: adjusted to balance dilation and preserve spatial size

### 9) `cifar10_v7.ipynb`
- Finalized an optimized `ciferNet` with the following design:
  - 3×3 kernels early for efficiency
  - Strides of 2 for controlled downsampling
  - Dilation ×2 in mid/late layers to expand receptive field
  - Depthwise separable convolutions in deeper layers to cut parameters
  - Balanced filter counts to hit ≈200k params
- Targeted outcomes:
  - Receptive Field ≈ 44
  - Parameter Count ≈ 200k

---

## Modular Code Layout

The final notebook (`cifar10_v7.ipynb`) has been refactored into a clean Python module structure so you can train directly from `main.py`:

```
models/
  cifernet.py        # ciferNet architecture (depthwise separable + dilation)
data/
  transforms.py      # Albumentations pipelines (train/test)
  dataset.py         # CIFAR-10 wrapper using Albumentations
  loader.py          # get_loaders(): returns train/test DataLoaders
engine/
  train.py           # train() and test() loops
main.py              # Entry point to run training/testing
```

---

## How to Run

1) Install dependencies:
```bash
pip install torch torchvision albumentations tqdm
```

2) Train the model:
```bash
python main.py
```

Defaults: SGD (lr=0.01, momentum=0.9), 35 epochs, Albumentations-enabled loaders, device auto-detected (CUDA if available).

---

## Notes and Next Steps
- Swap optimizers/schedulers easily in `main.py` (e.g., Adam + CosineAnnealingLR from the notebooks).
- Tune augmentations in `data/transforms.py` (e.g., probabilities, magnitudes).
- Consider label smoothing, CutMix/MixUp, or LR warmup if pushing beyond ~84%.

---

## Final Training Output (from `cifar10_v7.ipynb`)

```text
Training for 100 epochs...
EPOCH: 1
Loss=1.6004524230957031 Batch_id=390 Accuracy=42.85: 100%|██████████| 391/391 [00:21<00:00, 18.21it/s]

Test set: Average loss: 1.3343, Accuracy: 5226/10000 (52.26%)

EPOCH: 2
Loss=1.259232759475708 Batch_id=390 Accuracy=52.36: 100%|██████████| 391/391 [00:21<00:00, 18.46it/s]

Test set: Average loss: 1.1727, Accuracy: 5836/10000 (58.36%)

EPOCH: 3
Loss=1.2260401248931885 Batch_id=390 Accuracy=56.73: 100%|██████████| 391/391 [00:20<00:00, 19.20it/s]

Test set: Average loss: 1.0818, Accuracy: 6210/10000 (62.10%)

EPOCH: 4
Loss=1.1713800430297852 Batch_id=390 Accuracy=59.38: 100%|██████████| 391/391 [00:22<00:00, 17.67it/s]

Test set: Average loss: 1.0484, Accuracy: 6304/10000 (63.04%)

EPOCH: 5
Loss=1.1970027685165405 Batch_id=390 Accuracy=61.60: 100%|██████████| 391/391 [00:21<00:00, 18.47it/s]

Test set: Average loss: 0.9415, Accuracy: 6729/10000 (67.29%)

EPOCH: 6
Loss=0.9255263209342957 Batch_id=390 Accuracy=62.86: 100%|██████████| 391/391 [00:21<00:00, 18.46it/s]

Test set: Average loss: 0.9224, Accuracy: 6760/10000 (67.60%)

EPOCH: 7
Loss=1.1100846529006958 Batch_id=390 Accuracy=64.41: 100%|██████████| 391/391 [00:19<00:00, 19.61it/s]

Test set: Average loss: 0.9044, Accuracy: 6827/10000 (68.27%)

EPOCH: 8
Loss=0.8437091708183289 Batch_id=390 Accuracy=65.75: 100%|██████████| 391/391 [00:20<00:00, 19.42it/s]

Test set: Average loss: 0.8970, Accuracy: 6906/10000 (69.06%)

EPOCH: 9
Loss=0.9873706698417664 Batch_id=390 Accuracy=67.08: 100%|██████████| 391/391 [00:21<00:00, 18.51it/s]

Test set: Average loss: 0.8337, Accuracy: 7087/10000 (70.87%)

EPOCH: 10
Loss=1.1063858270645142 Batch_id=390 Accuracy=68.16: 100%|██████████| 391/391 [00:21<00:00, 18.45it/s]

Test set: Average loss: 0.7733, Accuracy: 7305/10000 (73.05%)

EPOCH: 11
Loss=0.7254306674003601 Batch_id=390 Accuracy=69.66: 100%|██████████| 391/391 [00:20<00:00, 18.84it/s]

Test set: Average loss: 0.7292, Accuracy: 7446/10000 (74.46%)
```

> Note: The above is a representative excerpt from the final training cell to showcase steady improvement; full logs are available in `cifar10_v7.ipynb`.

---

## Credits
- CIFAR-10 dataset and PyTorch ecosystem
- Albumentations for robust image augmentation

