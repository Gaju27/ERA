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
### Reached target test accuracy of 85.0% at epoch 81!

```text
Training for 100 epochs...
EPOCH: 1
Loss=0.4819863736629486 Batch_id=390 Accuracy=83.57: 100%|██████████| 391/391 [00:21<00:00, 18.21it/s]

Test set: Average loss: 0.4636, Accuracy: 8384/10000 (83.84%)

EPOCH: 2
Loss=0.4219619631767273 Batch_id=390 Accuracy=83.78: 100%|██████████| 391/391 [00:21<00:00, 18.25it/s]

Test set: Average loss: 0.4634, Accuracy: 8393/10000 (83.93%)

EPOCH: 3
Loss=0.5344582796096802 Batch_id=390 Accuracy=83.66: 100%|██████████| 391/391 [00:20<00:00, 19.14it/s]

Test set: Average loss: 0.4625, Accuracy: 8395/10000 (83.95%)

EPOCH: 4
Loss=0.638694167137146 Batch_id=390 Accuracy=83.55: 100%|██████████| 391/391 [00:20<00:00, 19.27it/s]

Test set: Average loss: 0.4662, Accuracy: 8386/10000 (83.86%)

EPOCH: 5
Loss=0.4234585762023926 Batch_id=390 Accuracy=83.16: 100%|██████████| 391/391 [00:21<00:00, 18.06it/s]

Test set: Average loss: 0.4745, Accuracy: 8366/10000 (83.66%)

EPOCH: 6
Loss=0.5753031969070435 Batch_id=390 Accuracy=82.35: 100%|██████████| 391/391 [00:21<00:00, 18.08it/s]

Test set: Average loss: 0.4827, Accuracy: 8330/10000 (83.30%)

EPOCH: 7
Loss=0.39405956864356995 Batch_id=390 Accuracy=81.45: 100%|██████████| 391/391 [00:20<00:00, 19.42it/s]

Test set: Average loss: 0.4946, Accuracy: 8297/10000 (82.97%)

EPOCH: 8
Loss=0.4324347972869873 Batch_id=390 Accuracy=80.22: 100%|██████████| 391/391 [00:20<00:00, 18.95it/s]

Test set: Average loss: 0.5315, Accuracy: 8225/10000 (82.25%)

EPOCH: 9
Loss=0.7712709307670593 Batch_id=390 Accuracy=79.39: 100%|██████████| 391/391 [00:21<00:00, 18.22it/s]

Test set: Average loss: 0.5573, Accuracy: 8086/10000 (80.86%)

EPOCH: 10
Loss=0.7484618425369263 Batch_id=390 Accuracy=78.23: 100%|██████████| 391/391 [00:20<00:00, 18.67it/s]

Test set: Average loss: 0.5598, Accuracy: 8096/10000 (80.96%)

EPOCH: 11
Loss=0.5484182834625244 Batch_id=390 Accuracy=77.24: 100%|██████████| 391/391 [00:20<00:00, 18.88it/s]

Test set: Average loss: 0.5775, Accuracy: 8010/10000 (80.10%)

EPOCH: 12
Loss=0.6049688458442688 Batch_id=390 Accuracy=76.60: 100%|██████████| 391/391 [00:21<00:00, 18.38it/s]

Test set: Average loss: 0.5908, Accuracy: 7987/10000 (79.87%)

EPOCH: 13
Loss=0.7310870885848999 Batch_id=390 Accuracy=75.94: 100%|██████████| 391/391 [00:21<00:00, 18.10it/s]

Test set: Average loss: 0.6486, Accuracy: 7741/10000 (77.41%)

EPOCH: 14
Loss=0.613690972328186 Batch_id=390 Accuracy=75.43: 100%|██████████| 391/391 [00:20<00:00, 19.00it/s]

Test set: Average loss: 0.6513, Accuracy: 7744/10000 (77.44%)

EPOCH: 15
Loss=0.6076458096504211 Batch_id=390 Accuracy=74.87: 100%|██████████| 391/391 [00:20<00:00, 18.91it/s]

Test set: Average loss: 0.6369, Accuracy: 7781/10000 (77.81%)

EPOCH: 16
Loss=0.6256329417228699 Batch_id=390 Accuracy=74.49: 100%|██████████| 391/391 [00:21<00:00, 17.91it/s]

Test set: Average loss: 0.6621, Accuracy: 7645/10000 (76.45%)

EPOCH: 17
Loss=0.888187050819397 Batch_id=390 Accuracy=73.73: 100%|██████████| 391/391 [00:21<00:00, 18.50it/s]

Test set: Average loss: 0.6774, Accuracy: 7658/10000 (76.58%)

EPOCH: 18
Loss=0.7820812463760376 Batch_id=390 Accuracy=73.61: 100%|██████████| 391/391 [00:19<00:00, 19.57it/s]

Test set: Average loss: 0.6700, Accuracy: 7689/10000 (76.89%)

EPOCH: 19
Loss=0.7872449159622192 Batch_id=390 Accuracy=73.28: 100%|██████████| 391/391 [00:21<00:00, 18.18it/s]

Test set: Average loss: 0.7298, Accuracy: 7484/10000 (74.84%)

EPOCH: 20
Loss=1.044542908668518 Batch_id=390 Accuracy=73.05: 100%|██████████| 391/391 [00:21<00:00, 18.38it/s]

Test set: Average loss: 0.6530, Accuracy: 7715/10000 (77.15%)

EPOCH: 21
Loss=0.7800201177597046 Batch_id=390 Accuracy=72.58: 100%|██████████| 391/391 [00:20<00:00, 19.17it/s]

Test set: Average loss: 0.6601, Accuracy: 7735/10000 (77.35%)

EPOCH: 22
Loss=0.7284479737281799 Batch_id=390 Accuracy=73.01: 100%|██████████| 391/391 [00:20<00:00, 18.74it/s]

Test set: Average loss: 0.6929, Accuracy: 7594/10000 (75.94%)

EPOCH: 23
Loss=0.8852490186691284 Batch_id=390 Accuracy=73.01: 100%|██████████| 391/391 [00:21<00:00, 17.97it/s]

Test set: Average loss: 0.7170, Accuracy: 7525/10000 (75.25%)

EPOCH: 24
Loss=0.7111740112304688 Batch_id=390 Accuracy=73.02: 100%|██████████| 391/391 [00:21<00:00, 18.02it/s]

Test set: Average loss: 0.6759, Accuracy: 7676/10000 (76.76%)

EPOCH: 25
Loss=0.9621587991714478 Batch_id=390 Accuracy=73.08: 100%|██████████| 391/391 [00:20<00:00, 18.95it/s]

Test set: Average loss: 0.7019, Accuracy: 7575/10000 (75.75%)

EPOCH: 26
Loss=0.6732138991355896 Batch_id=390 Accuracy=73.57: 100%|██████████| 391/391 [00:21<00:00, 18.37it/s]

Test set: Average loss: 0.6256, Accuracy: 7844/10000 (78.44%)

EPOCH: 27
Loss=0.7098684310913086 Batch_id=390 Accuracy=74.50: 100%|██████████| 391/391 [00:21<00:00, 17.96it/s]

Test set: Average loss: 0.6318, Accuracy: 7826/10000 (78.26%)

EPOCH: 28
Loss=0.9817219972610474 Batch_id=390 Accuracy=74.71: 100%|██████████| 391/391 [00:20<00:00, 18.80it/s]

Test set: Average loss: 0.6524, Accuracy: 7748/10000 (77.48%)

EPOCH: 29
Loss=0.625451385974884 Batch_id=390 Accuracy=75.49: 100%|██████████| 391/391 [00:20<00:00, 18.83it/s]

Test set: Average loss: 0.5974, Accuracy: 7955/10000 (79.55%)

EPOCH: 30
Loss=0.6327195763587952 Batch_id=390 Accuracy=76.14: 100%|██████████| 391/391 [00:21<00:00, 18.12it/s]

Test set: Average loss: 0.5848, Accuracy: 7958/10000 (79.58%)

EPOCH: 31
Loss=0.6073415875434875 Batch_id=390 Accuracy=76.97: 100%|██████████| 391/391 [00:21<00:00, 17.95it/s]

Test set: Average loss: 0.5698, Accuracy: 8022/10000 (80.22%)

EPOCH: 32
Loss=0.5162347555160522 Batch_id=390 Accuracy=78.00: 100%|██████████| 391/391 [00:20<00:00, 19.25it/s]

Test set: Average loss: 0.5515, Accuracy: 8057/10000 (80.57%)

EPOCH: 33
Loss=0.7489534616470337 Batch_id=390 Accuracy=78.70: 100%|██████████| 391/391 [00:20<00:00, 18.68it/s]

Test set: Average loss: 0.5317, Accuracy: 8156/10000 (81.56%)

EPOCH: 34
Loss=0.5036865472793579 Batch_id=390 Accuracy=79.82: 100%|██████████| 391/391 [00:21<00:00, 18.42it/s]

Test set: Average loss: 0.5156, Accuracy: 8239/10000 (82.39%)

EPOCH: 35
Loss=0.5402060747146606 Batch_id=390 Accuracy=80.78: 100%|██████████| 391/391 [00:20<00:00, 18.81it/s]

Test set: Average loss: 0.5049, Accuracy: 8250/10000 (82.50%)

EPOCH: 36
Loss=0.5641905069351196 Batch_id=390 Accuracy=82.13: 100%|██████████| 391/391 [00:20<00:00, 18.94it/s]

Test set: Average loss: 0.4851, Accuracy: 8318/10000 (83.18%)

EPOCH: 37
Loss=0.5495480298995972 Batch_id=390 Accuracy=82.74: 100%|██████████| 391/391 [00:21<00:00, 18.26it/s]

Test set: Average loss: 0.4693, Accuracy: 8397/10000 (83.97%)

EPOCH: 38
Loss=0.4815453886985779 Batch_id=390 Accuracy=83.51: 100%|██████████| 391/391 [00:21<00:00, 17.94it/s]

Test set: Average loss: 0.4595, Accuracy: 8392/10000 (83.92%)

EPOCH: 39
Loss=0.3935025632381439 Batch_id=390 Accuracy=84.09: 100%|██████████| 391/391 [00:21<00:00, 18.55it/s]

Test set: Average loss: 0.4565, Accuracy: 8409/10000 (84.09%)

EPOCH: 40
Loss=0.4513435363769531 Batch_id=390 Accuracy=84.59: 100%|██████████| 391/391 [00:21<00:00, 18.59it/s]

Test set: Average loss: 0.4503, Accuracy: 8443/10000 (84.43%)

EPOCH: 41
Loss=0.44021177291870117 Batch_id=390 Accuracy=84.51: 100%|██████████| 391/391 [00:21<00:00, 17.94it/s]

Test set: Average loss: 0.4523, Accuracy: 8446/10000 (84.46%)

EPOCH: 42
Loss=0.4537960886955261 Batch_id=390 Accuracy=84.51: 100%|██████████| 391/391 [00:21<00:00, 18.01it/s]

Test set: Average loss: 0.4512, Accuracy: 8440/10000 (84.40%)

EPOCH: 43
Loss=0.6112302541732788 Batch_id=390 Accuracy=84.59: 100%|██████████| 391/391 [00:20<00:00, 18.71it/s]

Test set: Average loss: 0.4542, Accuracy: 8433/10000 (84.33%)

EPOCH: 44
Loss=0.42850571870803833 Batch_id=390 Accuracy=84.22: 100%|██████████| 391/391 [00:20<00:00, 19.01it/s]

Test set: Average loss: 0.4554, Accuracy: 8436/10000 (84.36%)

EPOCH: 45
Loss=0.5978809595108032 Batch_id=390 Accuracy=83.52: 100%|██████████| 391/391 [00:22<00:00, 17.76it/s]

Test set: Average loss: 0.4643, Accuracy: 8405/10000 (84.05%)

EPOCH: 46
Loss=0.517680823802948 Batch_id=390 Accuracy=82.85: 100%|██████████| 391/391 [00:21<00:00, 18.18it/s]

Test set: Average loss: 0.4779, Accuracy: 8379/10000 (83.79%)

EPOCH: 47
Loss=0.5325216054916382 Batch_id=390 Accuracy=82.37: 100%|██████████| 391/391 [00:20<00:00, 19.00it/s]

Test set: Average loss: 0.4953, Accuracy: 8329/10000 (83.29%)

EPOCH: 48
Loss=0.7170684933662415 Batch_id=390 Accuracy=81.16: 100%|██████████| 391/391 [00:20<00:00, 19.18it/s]

Test set: Average loss: 0.5121, Accuracy: 8213/10000 (82.13%)

EPOCH: 49
Loss=0.707506000995636 Batch_id=390 Accuracy=79.99: 100%|██████████| 391/391 [00:21<00:00, 18.10it/s]

Test set: Average loss: 0.5223, Accuracy: 8205/10000 (82.05%)

EPOCH: 50
Loss=0.5667501091957092 Batch_id=390 Accuracy=78.98: 100%|██████████| 391/391 [00:21<00:00, 18.23it/s]

Test set: Average loss: 0.5497, Accuracy: 8147/10000 (81.47%)

EPOCH: 51
Loss=0.5945467948913574 Batch_id=390 Accuracy=78.29: 100%|██████████| 391/391 [00:20<00:00, 18.99it/s]

Test set: Average loss: 0.5622, Accuracy: 8053/10000 (80.53%)

EPOCH: 52
Loss=0.6600630879402161 Batch_id=390 Accuracy=77.56: 100%|██████████| 391/391 [00:20<00:00, 18.82it/s]

Test set: Average loss: 0.5982, Accuracy: 7933/10000 (79.33%)

EPOCH: 53
Loss=0.7022034525871277 Batch_id=390 Accuracy=76.75: 100%|██████████| 391/391 [00:21<00:00, 18.08it/s]

Test set: Average loss: 0.6143, Accuracy: 7942/10000 (79.42%)

EPOCH: 54
Loss=0.5662193894386292 Batch_id=390 Accuracy=75.86: 100%|██████████| 391/391 [00:21<00:00, 17.98it/s]

Test set: Average loss: 0.5834, Accuracy: 8023/10000 (80.23%)

EPOCH: 55
Loss=0.6941968202590942 Batch_id=390 Accuracy=75.36: 100%|██████████| 391/391 [00:20<00:00, 18.89it/s]

Test set: Average loss: 0.6456, Accuracy: 7748/10000 (77.48%)

EPOCH: 56
Loss=0.6810113787651062 Batch_id=390 Accuracy=74.86: 100%|██████████| 391/391 [00:21<00:00, 18.07it/s]

Test set: Average loss: 0.6930, Accuracy: 7678/10000 (76.78%)

EPOCH: 57
Loss=0.7034405469894409 Batch_id=390 Accuracy=74.55: 100%|██████████| 391/391 [00:21<00:00, 18.30it/s]

Test set: Average loss: 0.6555, Accuracy: 7769/10000 (77.69%)

EPOCH: 58
Loss=0.7027315497398376 Batch_id=390 Accuracy=74.25: 100%|██████████| 391/391 [00:21<00:00, 18.27it/s]

Test set: Average loss: 0.7301, Accuracy: 7548/10000 (75.48%)

EPOCH: 59
Loss=0.7924020886421204 Batch_id=390 Accuracy=73.66: 100%|██████████| 391/391 [00:20<00:00, 18.81it/s]

Test set: Average loss: 0.6764, Accuracy: 7670/10000 (76.70%)

EPOCH: 60
Loss=0.7558394074440002 Batch_id=390 Accuracy=73.85: 100%|██████████| 391/391 [00:22<00:00, 17.07it/s]

Test set: Average loss: 0.7276, Accuracy: 7491/10000 (74.91%)

EPOCH: 61
Loss=0.6460668444633484 Batch_id=390 Accuracy=73.48: 100%|██████████| 391/391 [00:22<00:00, 17.57it/s]

Test set: Average loss: 0.6470, Accuracy: 7777/10000 (77.77%)

EPOCH: 62
Loss=0.7880786657333374 Batch_id=390 Accuracy=73.43: 100%|██████████| 391/391 [00:21<00:00, 18.60it/s]

Test set: Average loss: 0.6489, Accuracy: 7743/10000 (77.43%)

EPOCH: 63
Loss=0.8190215826034546 Batch_id=390 Accuracy=73.95: 100%|██████████| 391/391 [00:20<00:00, 19.24it/s]

Test set: Average loss: 0.6453, Accuracy: 7776/10000 (77.76%)

EPOCH: 64
Loss=0.7102404832839966 Batch_id=390 Accuracy=73.73: 100%|██████████| 391/391 [00:21<00:00, 18.15it/s]

Test set: Average loss: 0.6507, Accuracy: 7752/10000 (77.52%)

EPOCH: 65
Loss=0.5750873684883118 Batch_id=390 Accuracy=73.93: 100%|██████████| 391/391 [00:21<00:00, 17.96it/s]

Test set: Average loss: 0.6478, Accuracy: 7797/10000 (77.97%)

EPOCH: 66
Loss=0.7313960194587708 Batch_id=390 Accuracy=74.30: 100%|██████████| 391/391 [00:21<00:00, 18.38it/s]

Test set: Average loss: 0.6450, Accuracy: 7828/10000 (78.28%)

EPOCH: 67
Loss=0.7456944584846497 Batch_id=390 Accuracy=74.65: 100%|██████████| 391/391 [00:20<00:00, 18.68it/s]

Test set: Average loss: 0.5993, Accuracy: 7913/10000 (79.13%)

EPOCH: 68
Loss=0.6166459321975708 Batch_id=390 Accuracy=75.61: 100%|██████████| 391/391 [00:22<00:00, 17.70it/s]

Test set: Average loss: 0.6394, Accuracy: 7835/10000 (78.35%)

EPOCH: 69
Loss=0.61135333776474 Batch_id=390 Accuracy=75.87: 100%|██████████| 391/391 [00:22<00:00, 17.70it/s]

Test set: Average loss: 0.6125, Accuracy: 7861/10000 (78.61%)

EPOCH: 70
Loss=0.9786127209663391 Batch_id=390 Accuracy=76.61: 100%|██████████| 391/391 [00:21<00:00, 18.01it/s]

Test set: Average loss: 0.6117, Accuracy: 7878/10000 (78.78%)

EPOCH: 71
Loss=0.7788032293319702 Batch_id=390 Accuracy=77.27: 100%|██████████| 391/391 [00:20<00:00, 18.90it/s]

Test set: Average loss: 0.5633, Accuracy: 8054/10000 (80.54%)

EPOCH: 72
Loss=0.622490644454956 Batch_id=390 Accuracy=78.25: 100%|██████████| 391/391 [00:21<00:00, 18.10it/s]

Test set: Average loss: 0.5550, Accuracy: 8119/10000 (81.19%)

EPOCH: 73
Loss=0.5214658975601196 Batch_id=390 Accuracy=79.10: 100%|██████████| 391/391 [00:21<00:00, 17.96it/s]

Test set: Average loss: 0.5333, Accuracy: 8135/10000 (81.35%)

EPOCH: 74
Loss=0.5881436467170715 Batch_id=390 Accuracy=80.24: 100%|██████████| 391/391 [00:20<00:00, 18.84it/s]

Test set: Average loss: 0.4886, Accuracy: 8337/10000 (83.37%)

EPOCH: 75
Loss=0.5150014758110046 Batch_id=390 Accuracy=81.09: 100%|██████████| 391/391 [00:21<00:00, 18.46it/s]

Test set: Average loss: 0.4990, Accuracy: 8302/10000 (83.02%)

EPOCH: 76
Loss=0.5216653943061829 Batch_id=390 Accuracy=82.21: 100%|██████████| 391/391 [00:22<00:00, 17.20it/s]

Test set: Average loss: 0.4796, Accuracy: 8370/10000 (83.70%)

EPOCH: 77
Loss=0.4745190739631653 Batch_id=390 Accuracy=82.98: 100%|██████████| 391/391 [00:22<00:00, 17.21it/s]

Test set: Average loss: 0.4639, Accuracy: 8432/10000 (84.32%)

EPOCH: 78
Loss=0.3954485356807709 Batch_id=390 Accuracy=83.87: 100%|██████████| 391/391 [00:21<00:00, 17.82it/s]

Test set: Average loss: 0.4562, Accuracy: 8468/10000 (84.68%)

EPOCH: 79
Loss=0.5522920489311218 Batch_id=390 Accuracy=84.55: 100%|██████████| 391/391 [00:21<00:00, 18.59it/s]

Test set: Average loss: 0.4517, Accuracy: 8485/10000 (84.85%)

EPOCH: 80
Loss=0.3613813817501068 Batch_id=390 Accuracy=84.59: 100%|██████████| 391/391 [00:22<00:00, 17.58it/s]

Test set: Average loss: 0.4471, Accuracy: 8476/10000 (84.76%)

EPOCH: 81
Loss=0.5563873052597046 Batch_id=390 Accuracy=84.56: 100%|██████████| 391/391 [00:22<00:00, 17.17it/s]

Test set: Average loss: 0.4473, Accuracy: 8503/10000 (85.03%)

Reached target test accuracy of 85.0% at epoch 81!

```

> Note: The above is a representative excerpt from the final training cell to showcase steady improvement; full logs are available in `cifar10_v7.ipynb`.

---

## Credits
- CIFAR-10 dataset and PyTorch ecosystem
- Albumentations for robust image augmentation


