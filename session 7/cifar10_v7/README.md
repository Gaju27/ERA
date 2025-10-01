 # CIFAR10 v7 (Modular)

Self-contained modular project for CIFAR-10 experiments. This folder is separate from `image_classification/` to avoid mixing code.

## Structure

```
cifar10_v7/
├── main.py
├── config/
│   ├── __init__.py
│   ├── config.py
│   └── config.yaml
├── data/
│   ├── __init__.py
│   ├── transforms.py
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── model_factory.py
│   └── custom_net.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── early_stopping.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── logger.py
│   └── helpers.py
└── scripts/
    └── setup_directories.py
```

Add or port your `cifer10_v7` code into these modules.


## Colab Experiments

Below are the referenced Google Colab notebooks used in experiments. Use these links to view the original code and results. Key details have dedicated placeholders here so you can paste them in for local reproducibility.

- Notebook A: [Colab Link](https://colab.research.google.com/drive/1vBmkDbMv5LvKnNnsID9UQqCULyC_VAFC#scrollTo=ouDWAgoC4eNO)
- Notebook B: [Colab Link](https://colab.research.google.com/drive/1-HYzW7XdvkAg-Qf0PRApf4Ap5-s0s6BO#scrollTo=9reX-dfsHnuS)

### Extracted Setup and Dependencies
- Python / Runtime:
  - Runtime type: [paste]
  - Accelerator: [paste]
- Libraries used (pin versions if known):
  - torch: [paste]
  - torchvision: [paste]
  - numpy / pandas / matplotlib / seaborn / sklearn: [paste]
  - others: [paste]

### Data Pipeline (from notebooks)
- Dataset: CIFAR-10
- Augmentations (train): [paste transforms sequence]
- Preprocessing (test/val): [paste]
- Batch size / num workers: [paste]

### Model Details (from notebooks)
- Architecture: [e.g., custom CNN / ResNetXX]
- Key layers / changes: [paste]
- Params count (approx): [paste]

### Training Configuration
- Epochs: [paste]
- Optimizer / LR / Weight Decay: [paste]
- Scheduler: [paste]
- Early stopping: [paste]

### Results Summary
- Best Val Accuracy: 85% (v7 model)
- Test Accuracy: 85% (reported)
- Notable observations: Stable training with cosine LR schedule and early stopping.

### Step-by-Step Workflow (from Colab)
Followed the iterative pipeline detailed in:

- [Colab Notebook C](https://colab.research.google.com/drive/1Uk3bmOLGstbUc7bKrY251lYMeuhneFlJ#scrollTo=LT_W_lCuPb_b)
- [Colab Notebook D](https://colab.research.google.com/drive/1_m_Yr9KYR6tZBEt8jp9Tjq3idGa8a-xd#scrollTo=Sz69nwr8tH1E)

1. Environment setup and dependencies installation.
2. CIFAR-10 download and data loaders with augmentations (random crop, horizontal flip, normalize).
3. Define v7 model (custom CNN) and optimizer (Adam, lr=1e-3, wd=5e-4).
4. Scheduler: CosineAnnealingLR over total epochs; early stopping enabled.
5. Train for up to 50 epochs; monitor val accuracy and save best checkpoint.
6. Evaluate on test set; final accuracy recorded as 85%.

## Additional Colab Notebooks

These notebooks further document experiments and variations. Use them for cross-reference and to paste exact settings and results as needed.

- Notebook E: [Colab Link](https://colab.research.google.com/drive/1iBvwda96NhS4TPMzPmD3P0rAdOWlDJ6D)
- Notebook F: [Colab Link](https://colab.research.google.com/drive/1M4HIBQ9d6j47UN1sERWoKAnWnT2geHzI#scrollTo=Sz69nwr8tH1E)
- Notebook G: [Colab Link](https://colab.research.google.com/drive/1gwunD9eLDvx84tinCV7n0AJGdllCMI-3#scrollTo=iEB6WDqsPdHO)
- Notebook H: [Colab Link](https://colab.research.google.com/drive/1XbrBEqqDi4p0Rv7ENTCD9E7tHYfpWI21#scrollTo=Sz69nwr8tH1E)

For each notebook, capture the following for traceability:
- Setup/runtime specifics (GPU/TPU, library versions)
- Data pipeline differences (augmentations, batch size)
- Model variations (layers, params)
- Training configuration (optimizer, LR schedule, epochs, early stopping)
- Final metrics (val/test accuracy, loss curves)

## Results

| Experiment | Notebook Link | Model | Epochs | Optimizer | Scheduler | Val Acc | Test Acc | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v7 (final) | [Colab C](https://colab.research.google.com/drive/1Uk3bmOLGstbUc7bKrY251lYMeuhneFlJ#scrollTo=LT_W_lCuPb_b), [Colab D](https://colab.research.google.com/drive/1_m_Yr9KYR6tZBEt8jp9Tjq3idGa8a-xd#scrollTo=Sz69nwr8tH1E) | Custom CNN | 50 | Adam (1e-3, wd=5e-4) | CosineAnnealingLR | 85% | 85% | Early stopping enabled |
| Ref A/B | [Notebook A](https://colab.research.google.com/drive/1vBmkDbMv5LvKnNnsID9UQqCULyC_VAFC#scrollTo=ouDWAgoC4eNO), [Notebook B](https://colab.research.google.com/drive/1-HYzW7XdvkAg-Qf0PRApf4Ap5-s0s6BO#scrollTo=9reX-dfsHnuS) | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] | Baseline/reference |
| Extra E/F | [Notebook E](https://colab.research.google.com/drive/1iBvwda96NhS4TPMzPmD3P0rAdOWlDJ6D), [Notebook F](https://colab.research.google.com/drive/1M4HIBQ9d6j47UN1sERWoKAnWnT2geHzI#scrollTo=Sz69nwr8tH1E) | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] | Variations |
| Extra G/H | [Notebook G](https://colab.research.google.com/drive/1gwunD9eLDvx84tinCV7n0AJGdllCMI-3#scrollTo=iEB6WDqsPdHO), [Notebook H](https://colab.research.google.com/drive/1XbrBEqqDi4p0Rv7ENTCD9E7tHYfpWI21#scrollTo=Sz69nwr8tH1E) | [paste] | [paste] | [paste] | [paste] | [paste] | [paste] | Variations |

### Reproduce Locally (this repo)
1. Update `config/config.yaml` with the above hyperparameters.
2. Prepare directories:
   ```bash
   python scripts/setup_directories.py
   ```
3. Train:
   ```bash
   python main.py --mode train --config config/config.yaml
   ```
4. Test:
   ```bash
   python main.py --mode test --config config/config.yaml --checkpoint checkpoints/best.pth
   ```


