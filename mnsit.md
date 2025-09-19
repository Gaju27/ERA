# ConvNet – PyTorch CNN for MNIST

## Overview
This project implements a **Convolutional Neural Network (CNN)** using PyTorch to classify handwritten digits (MNIST dataset).  
The model is lightweight (~10k parameters) but still achieves **high accuracy (>99%)** with proper training.  

You can see the full implementation & training notebook here:  
[MNSIT Notebook](https://github.com/Gaju27/ERA/blob/main/Pytorch_tutorial.ipynb)

---

## Model Architecture
The network is defined in `ConvNet(nn.Module)`:

1. **Convolutional Layers**
   - `Conv2d(1 → 16, kernel=3, padding=1)` + BatchNorm + ReLU + MaxPool  
   - `Conv2d(16 → 32, kernel=3, padding=1)` + BatchNorm + ReLU + MaxPool  
   - `Conv2d(32 → 40, kernel=3, padding=1)` + BatchNorm + ReLU  

2. **Pooling**
   - `MaxPool2d(2,2)` for downsampling.  
   - `AdaptiveAvgPool2d(1,1)` for Global Average Pooling (reduces spatial dims to `1x1`).  

3. **Regularization**
   - Dropout (`p=0.3`) applied before the final layer to prevent overfitting.  

4. **Fully Connected Layer**
   - `Linear(40 → 10)` mapping GAP output to 10 digit classes.  

---

## Forward Pass
1. Input → Conv → BatchNorm → ReLU → MaxPool  
2. Conv → BatchNorm → ReLU → MaxPool  
3. Conv → BatchNorm → ReLU  
4. Global Average Pooling → Flatten → Dropout  
5. Final FC Layer → Output logits  

---

## Training Setup
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** SGD / Adam (configurable)  
- **Scheduler:** StepLR or ReduceLROnPlateau (recommended)  
- **Batch Size:** 64 / 128  
- **Epochs:** 15–20 usually sufficient  

---

## Results
- Compact model with ~**10k parameters**.  
- Reaches **>99% accuracy** on MNIST in fewer than 20 epochs.  
- Balanced between performance and efficiency.  

---

## Key Takeaways
- **Batch Normalization** improves training stability.  
- **Global Average Pooling** reduces parameters compared to fully connected hidden layers.  
- **Dropout** adds robustness and prevents overfitting.  
- Architecture is modular and can be easily extended.  
