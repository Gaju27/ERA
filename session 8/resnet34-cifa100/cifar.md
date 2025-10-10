
# 🧠 ResNet from Scratch on CIFAR-100

## 📘 Project Overview

This project trains a **ResNet model from scratch** on the **CIFAR-100 dataset** using PyTorch in Google Colab.  
The objective is to achieve **at least 73% Top-1 accuracy** (without using a pre-trained model).  
All training metrics and checkpoints are saved for reproducibility.

---

## ⚙️ 1. Setup Environment

Run this block to set up your environment in Colab:

```bash
!pip install torch torchvision torchaudio tqdm tensorboard
````

---

## 📂 2. Directory Structure

After running the notebook, you will see these folders in the left panel:

```
checkpoints/
 └── best_model.pth             # best model weights
data/
 └── cifar-100-python/          # dataset auto-downloaded by torchvision
logs/
 ├── train/                     # TensorBoard training logs
 └── val/                       # TensorBoard validation logs
sample_data/
```

---

## 🧩 3. Model — ResNet Architecture

The model used is a **custom ResNet** built from scratch with:

* Basic residual blocks (`conv → batchnorm → ReLU → skip connection`)
* Configurable number of layers (e.g., ResNet-18)
* No pretrained weights (initialized with Kaiming normal)

The code defines:

```python
class BasicBlock(nn.Module)
class ResNet(nn.Module)
```

---

## 📦 4. Data Preparation

CIFAR-100 dataset is automatically downloaded and preprocessed using:

* **RandomCrop(32, padding=4)**
* **RandomHorizontalFlip()**
* **Normalization(mean, std)**

```python
train_dataset = datasets.CIFAR100(root='./data', train=True, transform=train_transform, download=True)
test_dataset  = datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
```

---

## 🚀 5. Training Configuration

| Parameter     | Value                                         |
| ------------- | --------------------------------------------- |
| Optimizer     | SGD (lr=0.1, momentum=0.9, weight_decay=5e-4) |
| LR Scheduler  | StepLR (step_size=30, gamma=0.1)              |
| Loss Function | CrossEntropyLoss                              |
| Epochs        | 100                                           |
| Batch Size    | 128                                           |
| Device        | GPU (CUDA)                                    |

---

## 🧮 6. Accuracy Metrics

Two metrics are tracked:

* **Top-1 Accuracy:** Correct top prediction matches target label
* **Top-5 Accuracy:** Target label appears in top-5 predictions

Both are printed after every validation epoch and logged in TensorBoard.

---

## 🏋️ 7. Logging & Visualization

TensorBoard logs are stored in:

```
logs/train/
logs/val/
```

To visualize them in Colab:

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

---

## 💾 8. Checkpointing and Best Model Saving

Each epoch saves the model if it achieves a **new best validation accuracy**:

```python
if val_acc > best_acc:
    best_acc = val_acc
    torch.save(model.state_dict(), 'checkpoints/best_model.pth')
```

You can resume training later with:

```python
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
```

---

## 📊 9. Evaluation

After training:

```python
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
top1_acc, top5_acc = evaluate(model, test_loader)
print(f"✅ Final Top-1 Accuracy: {top1_acc:.2f}%")
print(f"✅ Final Top-5 Accuracy: {top5_acc:.2f}%")
```

---

## 📦 10. Exporting Your Work

Once training is complete, zip and download all logs and checkpoints:

```python
from google.colab import files
!zip -r project_files.zip checkpoints logs
files.download('project_files.zip')
```

---

## 🧠 11. Expected Performance

| Model     | Epochs | Approx. Top-1 Accuracy | Comments                      |
| --------- | ------ | ---------------------- | ----------------------------- |
| ResNet-18 | 100    | 72–73%                 | Baseline                      |
| ResNet-34 | 100    | 73–75%                 | Better depth, slower training |

---

## 🧩 12. Logs

```
Epoch 1/200  Time:45.6s  TrainLoss:4.2906  TrainTop1:5.03  ValLoss:4.0930  ValTop1:8.66  Best:8.66
Epoch 2/200  Time:43.3s  TrainLoss:3.9310  TrainTop1:10.78  ValLoss:3.8263  ValTop1:14.30  Best:14.30
Epoch 3/200  Time:45.1s  TrainLoss:3.6315  TrainTop1:17.11  ValLoss:3.4870  ValTop1:20.49  Best:20.49
Epoch 4/200  Time:44.0s  TrainLoss:3.2878  TrainTop1:25.18  ValLoss:3.3830  ValTop1:24.17  Best:24.17
Epoch 5/200  Time:45.1s  TrainLoss:3.0270  TrainTop1:31.62  ValLoss:3.1040  ValTop1:30.56  Best:30.56
Epoch 6/200  Time:44.0s  TrainLoss:2.7992  TrainTop1:37.75  ValLoss:3.0126  ValTop1:34.15  Best:34.15
Epoch 7/200  Time:44.9s  TrainLoss:2.6523  TrainTop1:41.93  ValLoss:2.9659  ValTop1:36.08  Best:36.08
Epoch 8/200  Time:43.9s  TrainLoss:2.5557  TrainTop1:44.62  ValLoss:2.6424  ValTop1:41.44  Best:41.44
Epoch 9/200  Time:44.5s  TrainLoss:2.4781  TrainTop1:46.83  ValLoss:2.5966  ValTop1:44.64  Best:44.64
Epoch 10/200  Time:43.8s  TrainLoss:2.4248  TrainTop1:48.49  ValLoss:2.7444  ValTop1:41.55  Best:44.64
Epoch 11/200  Time:44.8s  TrainLoss:2.3732  TrainTop1:50.08  ValLoss:2.6360  ValTop1:43.91  Best:44.64
Epoch 12/200  Time:43.9s  TrainLoss:2.3409  TrainTop1:50.80  ValLoss:2.8058  ValTop1:39.07  Best:44.64
Epoch 13/200  Time:44.6s  TrainLoss:2.3135  TrainTop1:51.97  ValLoss:2.8357  ValTop1:40.86  Best:44.64
Epoch 14/200  Time:43.8s  TrainLoss:2.2781  TrainTop1:53.31  ValLoss:2.7487  ValTop1:43.13  Best:44.64
Epoch 15/200  Time:44.5s  TrainLoss:2.2576  TrainTop1:53.56  ValLoss:2.4607  ValTop1:48.16  Best:48.16
Epoch 16/200  Time:44.2s  TrainLoss:2.2353  TrainTop1:54.50  ValLoss:2.5854  ValTop1:45.08  Best:48.16
Epoch 17/200  Time:44.6s  TrainLoss:2.2207  TrainTop1:54.66  ValLoss:2.9366  ValTop1:38.53  Best:48.16
Epoch 18/200  Time:44.0s  TrainLoss:2.2000  TrainTop1:55.24  ValLoss:2.5124  ValTop1:47.93  Best:48.16
Epoch 19/200  Time:44.5s  TrainLoss:2.1837  TrainTop1:56.00  ValLoss:2.8888  ValTop1:39.47  Best:48.16
Epoch 20/200  Time:43.9s  TrainLoss:2.1740  TrainTop1:56.36  ValLoss:2.5694  ValTop1:47.64  Best:48.16
Epoch 21/200  Time:44.9s  TrainLoss:2.1606  TrainTop1:56.55  ValLoss:2.5525  ValTop1:46.48  Best:48.16
Epoch 22/200  Time:44.0s  TrainLoss:2.1502  TrainTop1:56.81  ValLoss:2.4253  ValTop1:49.79  Best:49.79
Epoch 23/200  Time:44.6s  TrainLoss:2.1379  TrainTop1:57.07  ValLoss:2.5834  ValTop1:47.05  Best:49.79
Epoch 24/200  Time:44.2s  TrainLoss:2.1297  TrainTop1:57.35  ValLoss:2.4160  ValTop1:49.40  Best:49.79
Epoch 25/200  Time:44.7s  TrainLoss:2.1185  TrainTop1:57.79  ValLoss:2.5735  ValTop1:46.67  Best:49.79
Epoch 26/200  Time:44.2s  TrainLoss:2.1179  TrainTop1:58.08  ValLoss:2.7903  ValTop1:41.48  Best:49.79
Epoch 27/200  Time:44.6s  TrainLoss:2.1060  TrainTop1:58.12  ValLoss:2.5249  ValTop1:47.76  Best:49.79
Epoch 28/200  Time:47.1s  TrainLoss:2.0991  TrainTop1:58.23  ValLoss:2.4507  ValTop1:51.59  Best:51.59
Epoch 29/200  Time:44.4s  TrainLoss:2.0914  TrainTop1:58.48  ValLoss:2.5192  ValTop1:48.14  Best:51.59
Epoch 30/200  Time:44.6s  TrainLoss:2.0808  TrainTop1:59.03  ValLoss:2.4966  ValTop1:48.76  Best:51.59
Epoch 31/200  Time:44.7s  TrainLoss:2.0829  TrainTop1:58.81  ValLoss:2.4112  ValTop1:50.07  Best:51.59
Epoch 32/200  Time:44.3s  TrainLoss:2.0721  TrainTop1:59.35  ValLoss:2.4044  ValTop1:51.33  Best:51.59
Epoch 33/200  Time:44.8s  TrainLoss:2.0607  TrainTop1:59.57  ValLoss:2.3537  ValTop1:52.40  Best:52.40
Epoch 34/200  Time:44.1s  TrainLoss:2.0581  TrainTop1:59.99  ValLoss:2.3235  ValTop1:52.92  Best:52.92
Epoch 35/200  Time:44.6s  TrainLoss:2.0532  TrainTop1:59.83  ValLoss:2.4223  ValTop1:50.24  Best:52.92
Epoch 36/200  Time:44.1s  TrainLoss:2.0518  TrainTop1:59.78  ValLoss:2.4724  ValTop1:49.72  Best:52.92
Epoch 37/200  Time:44.5s  TrainLoss:2.0507  TrainTop1:60.07  ValLoss:2.3960  ValTop1:50.59  Best:52.92
Epoch 38/200  Time:43.9s  TrainLoss:2.0384  TrainTop1:60.34  ValLoss:2.5623  ValTop1:47.51  Best:52.92
Epoch 39/200  Time:44.7s  TrainLoss:2.0391  TrainTop1:60.33  ValLoss:2.4425  ValTop1:49.72  Best:52.92
Epoch 40/200  Time:43.9s  TrainLoss:2.0318  TrainTop1:60.33  ValLoss:2.3741  ValTop1:51.17  Best:52.92
Epoch 41/200  Time:44.5s  TrainLoss:2.0238  TrainTop1:60.84  ValLoss:2.4984  ValTop1:49.41  Best:52.92
Epoch 42/200  Time:44.0s  TrainLoss:2.0210  TrainTop1:61.01  ValLoss:2.3533  ValTop1:52.64  Best:52.92
Epoch 43/200  Time:44.8s  TrainLoss:2.0156  TrainTop1:61.07  ValLoss:2.7110  ValTop1:44.16  Best:52.92
Epoch 44/200  Time:44.4s  TrainLoss:2.0218  TrainTop1:60.66  ValLoss:2.5030  ValTop1:48.29  Best:52.92
Epoch 45/200  Time:45.0s  TrainLoss:2.0058  TrainTop1:61.30  ValLoss:2.3783  ValTop1:51.72  Best:52.92
Epoch 46/200  Time:44.3s  TrainLoss:2.0020  TrainTop1:61.25  ValLoss:2.4776  ValTop1:50.35  Best:52.92
Epoch 47/200  Time:44.4s  TrainLoss:1.9968  TrainTop1:61.71  ValLoss:2.4694  ValTop1:49.86  Best:52.92
Epoch 48/200  Time:44.1s  TrainLoss:2.0026  TrainTop1:61.13  ValLoss:2.2516  ValTop1:55.28  Best:55.28
Epoch 49/200  Time:44.4s  TrainLoss:1.9936  TrainTop1:61.63  ValLoss:2.4479  ValTop1:49.93  Best:55.28
Epoch 50/200  Time:44.0s  TrainLoss:1.9875  TrainTop1:61.94  ValLoss:2.2571  ValTop1:54.22  Best:55.28
Epoch 51/200  Time:44.4s  TrainLoss:1.9834  TrainTop1:61.84  ValLoss:2.2874  ValTop1:53.33  Best:55.28
Epoch 52/200  Time:43.9s  TrainLoss:1.9765  TrainTop1:62.33  ValLoss:2.4375  ValTop1:48.70  Best:55.28
Epoch 53/200  Time:44.2s  TrainLoss:1.9683  TrainTop1:62.16  ValLoss:2.3264  ValTop1:52.89  Best:55.28
Epoch 54/200  Time:43.8s  TrainLoss:1.9707  TrainTop1:62.32  ValLoss:2.4250  ValTop1:51.66  Best:55.28
Epoch 55/200  Time:44.3s  TrainLoss:1.9668  TrainTop1:62.37  ValLoss:2.3284  ValTop1:53.14  Best:55.28
Epoch 56/200  Time:43.8s  TrainLoss:1.9591  TrainTop1:62.86  ValLoss:2.4554  ValTop1:50.54  Best:55.28
Epoch 57/200  Time:44.6s  TrainLoss:1.9606  TrainTop1:62.72  ValLoss:2.4297  ValTop1:49.64  Best:55.28
Epoch 58/200  Time:43.7s  TrainLoss:1.9476  TrainTop1:62.99  ValLoss:2.2383  ValTop1:54.23  Best:55.28
Epoch 59/200  Time:44.8s  TrainLoss:1.9544  TrainTop1:62.79  ValLoss:2.2398  ValTop1:55.52  Best:55.52
Epoch 60/200  Time:43.9s  TrainLoss:1.9474  TrainTop1:63.04  ValLoss:2.2891  ValTop1:53.62  Best:55.52
Epoch 61/200  Time:44.6s  TrainLoss:1.9353  TrainTop1:63.38  ValLoss:2.2802  ValTop1:53.81  Best:55.52
Epoch 62/200  Time:43.9s  TrainLoss:1.9424  TrainTop1:63.06  ValLoss:2.5718  ValTop1:47.39  Best:55.52
Epoch 63/200  Time:45.0s  TrainLoss:1.9267  TrainTop1:63.61  ValLoss:2.3840  ValTop1:51.60  Best:55.52
Epoch 64/200  Time:45.7s  TrainLoss:1.9195  TrainTop1:63.97  ValLoss:2.2668  ValTop1:54.49  Best:55.52
Epoch 65/200  Time:44.2s  TrainLoss:1.9235  TrainTop1:63.79  ValLoss:2.3084  ValTop1:53.07  Best:55.52
Epoch 66/200  Time:44.1s  TrainLoss:1.9266  TrainTop1:63.63  ValLoss:2.2304  ValTop1:55.49  Best:55.52
Epoch 67/200  Time:44.3s  TrainLoss:1.9092  TrainTop1:64.19  ValLoss:2.4974  ValTop1:50.20  Best:55.52
Epoch 68/200  Time:44.1s  TrainLoss:1.9122  TrainTop1:64.16  ValLoss:2.2255  ValTop1:55.61  Best:55.61
Epoch 69/200  Time:44.2s  TrainLoss:1.9086  TrainTop1:64.44  ValLoss:2.2554  ValTop1:54.97  Best:55.61
Epoch 70/200  Time:44.0s  TrainLoss:1.8979  TrainTop1:64.74  ValLoss:2.3232  ValTop1:53.10  Best:55.61
Epoch 71/200  Time:44.0s  TrainLoss:1.9028  TrainTop1:64.34  ValLoss:2.4586  ValTop1:49.79  Best:55.61
Epoch 72/200  Time:43.9s  TrainLoss:1.8949  TrainTop1:64.71  ValLoss:2.3256  ValTop1:53.76  Best:55.61
Epoch 73/200  Time:44.2s  TrainLoss:1.8840  TrainTop1:65.11  ValLoss:2.3109  ValTop1:54.24  Best:55.61
Epoch 74/200  Time:43.8s  TrainLoss:1.8891  TrainTop1:64.67  ValLoss:2.2696  ValTop1:54.66  Best:55.61
Epoch 75/200  Time:44.2s  TrainLoss:1.8781  TrainTop1:65.25  ValLoss:2.1727  ValTop1:56.46  Best:56.46
Epoch 76/200  Time:43.8s  TrainLoss:1.8713  TrainTop1:65.60  ValLoss:2.3173  ValTop1:52.94  Best:56.46
Epoch 77/200  Time:44.6s  TrainLoss:1.8685  TrainTop1:65.53  ValLoss:2.3199  ValTop1:54.54  Best:56.46
Epoch 78/200  Time:44.0s  TrainLoss:1.8639  TrainTop1:65.72  ValLoss:2.2793  ValTop1:54.82  Best:56.46
Epoch 79/200  Time:44.7s  TrainLoss:1.8601  TrainTop1:65.77  ValLoss:2.2134  ValTop1:56.71  Best:56.71
Epoch 80/200  Time:43.9s  TrainLoss:1.8497  TrainTop1:66.11  ValLoss:2.1858  ValTop1:56.74  Best:56.74
Epoch 81/200  Time:44.6s  TrainLoss:1.8469  TrainTop1:66.09  ValLoss:2.2159  ValTop1:56.00  Best:56.74
Epoch 82/200  Time:44.0s  TrainLoss:1.8453  TrainTop1:66.33  ValLoss:2.2791  ValTop1:54.49  Best:56.74
Epoch 83/200  Time:44.6s  TrainLoss:1.8359  TrainTop1:66.65  ValLoss:2.2095  ValTop1:56.69  Best:56.74
Epoch 84/200  Time:43.9s  TrainLoss:1.8316  TrainTop1:66.92  ValLoss:2.1758  ValTop1:56.99  Best:56.99
Epoch 85/200  Time:44.6s  TrainLoss:1.8305  TrainTop1:66.74  ValLoss:2.3311  ValTop1:53.28  Best:56.99
Epoch 86/200  Time:44.0s  TrainLoss:1.8279  TrainTop1:66.77  ValLoss:2.2283  ValTop1:56.14  Best:56.99
Epoch 87/200  Time:44.9s  TrainLoss:1.8136  TrainTop1:67.23  ValLoss:2.1975  ValTop1:57.26  Best:57.26
Epoch 88/200  Time:43.9s  TrainLoss:1.8147  TrainTop1:67.22  ValLoss:2.2928  ValTop1:54.49  Best:57.26
Epoch 89/200  Time:44.8s  TrainLoss:1.8137  TrainTop1:67.17  ValLoss:2.1367  ValTop1:59.03  Best:59.03
Epoch 90/200  Time:43.8s  TrainLoss:1.8037  TrainTop1:67.35  ValLoss:2.2391  ValTop1:56.40  Best:59.03
Epoch 91/200  Time:44.6s  TrainLoss:1.7941  TrainTop1:67.83  ValLoss:2.3063  ValTop1:54.40  Best:59.03
Epoch 92/200  Time:43.9s  TrainLoss:1.7960  TrainTop1:67.78  ValLoss:2.2133  ValTop1:56.67  Best:59.03
Epoch 93/200  Time:44.2s  TrainLoss:1.7899  TrainTop1:68.01  ValLoss:2.2332  ValTop1:55.76  Best:59.03
Epoch 94/200  Time:44.2s  TrainLoss:1.7830  TrainTop1:68.17  ValLoss:2.1768  ValTop1:57.64  Best:59.03
Epoch 95/200  Time:44.0s  TrainLoss:1.7771  TrainTop1:68.43  ValLoss:2.3098  ValTop1:54.11  Best:59.03
Epoch 96/200  Time:44.2s  TrainLoss:1.7714  TrainTop1:68.59  ValLoss:2.2616  ValTop1:55.89  Best:59.03
Epoch 97/200  Time:43.8s  TrainLoss:1.7634  TrainTop1:68.92  ValLoss:2.2455  ValTop1:55.04  Best:59.03
Epoch 98/200  Time:44.2s  TrainLoss:1.7551  TrainTop1:69.01  ValLoss:2.2350  ValTop1:55.89  Best:59.03
Epoch 99/200  Time:43.7s  TrainLoss:1.7545  TrainTop1:68.87  ValLoss:2.1922  ValTop1:56.74  Best:59.03
Epoch 100/200  Time:44.2s  TrainLoss:1.7449  TrainTop1:69.40  ValLoss:2.2208  ValTop1:56.99  Best:59.03
Epoch 101/200  Time:43.8s  TrainLoss:1.7393  TrainTop1:69.54  ValLoss:2.0908  ValTop1:59.68  Best:59.68
Epoch 102/200  Time:44.6s  TrainLoss:1.7360  TrainTop1:69.65  ValLoss:2.1835  ValTop1:57.74  Best:59.68
Epoch 103/200  Time:44.0s  TrainLoss:1.7240  TrainTop1:69.99  ValLoss:2.1853  ValTop1:58.27  Best:59.68
Epoch 104/200  Time:45.1s  TrainLoss:1.7276  TrainTop1:69.98  ValLoss:2.0932  ValTop1:59.55  Best:59.68
Epoch 105/200  Time:43.6s  TrainLoss:1.7182  TrainTop1:70.28  ValLoss:2.1909  ValTop1:58.07  Best:59.68
Epoch 106/200  Time:45.1s  TrainLoss:1.7075  TrainTop1:70.52  ValLoss:2.1899  ValTop1:56.88  Best:59.68
Epoch 107/200  Time:43.7s  TrainLoss:1.7098  TrainTop1:70.68  ValLoss:2.1225  ValTop1:59.35  Best:59.68
Epoch 108/200  Time:44.6s  TrainLoss:1.6956  TrainTop1:71.06  ValLoss:2.2213  ValTop1:57.29  Best:59.68
Epoch 109/200  Time:43.8s  TrainLoss:1.6966  TrainTop1:70.86  ValLoss:2.0829  ValTop1:60.45  Best:60.45
Epoch 110/200  Time:44.4s  TrainLoss:1.6809  TrainTop1:71.47  ValLoss:2.1197  ValTop1:59.05  Best:60.45
Epoch 111/200  Time:44.2s  TrainLoss:1.6783  TrainTop1:71.43  ValLoss:2.0406  ValTop1:62.31  Best:62.31
Epoch 112/200  Time:44.1s  TrainLoss:1.6684  TrainTop1:72.10  ValLoss:2.0630  ValTop1:61.81  Best:62.31
Epoch 113/200  Time:44.3s  TrainLoss:1.6626  TrainTop1:72.04  ValLoss:2.1402  ValTop1:57.60  Best:62.31
Epoch 114/200  Time:44.3s  TrainLoss:1.6606  TrainTop1:72.14  ValLoss:2.0734  ValTop1:60.36  Best:62.31
Epoch 115/200  Time:44.5s  TrainLoss:1.6539  TrainTop1:72.28  ValLoss:2.0321  ValTop1:61.67  Best:62.31
Epoch 116/200  Time:44.1s  TrainLoss:1.6453  TrainTop1:72.41  ValLoss:2.0787  ValTop1:60.53  Best:62.31
Epoch 117/200  Time:44.4s  TrainLoss:1.6381  TrainTop1:72.64  ValLoss:2.1442  ValTop1:58.48  Best:62.31
Epoch 118/200  Time:44.0s  TrainLoss:1.6257  TrainTop1:73.18  ValLoss:2.2390  ValTop1:57.30  Best:62.31
Epoch 119/200  Time:44.4s  TrainLoss:1.6293  TrainTop1:73.16  ValLoss:2.0971  ValTop1:61.01  Best:62.31
Epoch 120/200  Time:44.0s  TrainLoss:1.6214  TrainTop1:73.27  ValLoss:2.1519  ValTop1:59.12  Best:62.31
Epoch 121/200  Time:44.0s  TrainLoss:1.6048  TrainTop1:73.91  ValLoss:2.0326  ValTop1:61.31  Best:62.31
Epoch 122/200  Time:43.6s  TrainLoss:1.5998  TrainTop1:73.99  ValLoss:2.0324  ValTop1:61.75  Best:62.31
Epoch 123/200  Time:44.3s  TrainLoss:1.5948  TrainTop1:74.34  ValLoss:2.1368  ValTop1:59.48  Best:62.31
Epoch 124/200  Time:43.9s  TrainLoss:1.5825  TrainTop1:74.65  ValLoss:2.0174  ValTop1:62.23  Best:62.31
Epoch 125/200  Time:44.5s  TrainLoss:1.5754  TrainTop1:74.77  ValLoss:2.0132  ValTop1:62.38  Best:62.38
Epoch 126/200  Time:43.7s  TrainLoss:1.5701  TrainTop1:74.98  ValLoss:2.1204  ValTop1:60.82  Best:62.38
Epoch 127/200  Time:44.5s  TrainLoss:1.5574  TrainTop1:75.35  ValLoss:2.0589  ValTop1:61.53  Best:62.38
Epoch 128/200  Time:43.7s  TrainLoss:1.5558  TrainTop1:75.58  ValLoss:1.9492  ValTop1:63.97  Best:63.97
Epoch 129/200  Time:44.3s  TrainLoss:1.5398  TrainTop1:76.16  ValLoss:1.9722  ValTop1:62.88  Best:63.97
Epoch 130/200  Time:43.7s  TrainLoss:1.5357  TrainTop1:76.23  ValLoss:1.9622  ValTop1:64.74  Best:64.74
Epoch 131/200  Time:44.2s  TrainLoss:1.5316  TrainTop1:76.59  ValLoss:2.0504  ValTop1:60.67  Best:64.74
Epoch 132/200  Time:44.2s  TrainLoss:1.5226  TrainTop1:76.55  ValLoss:2.0781  ValTop1:61.16  Best:64.74
Epoch 133/200  Time:43.7s  TrainLoss:1.5106  TrainTop1:77.11  ValLoss:2.0335  ValTop1:62.53  Best:64.74
Epoch 134/200  Time:46.5s  TrainLoss:1.5052  TrainTop1:77.15  ValLoss:1.9592  ValTop1:63.91  Best:64.74
Epoch 135/200  Time:44.1s  TrainLoss:1.4964  TrainTop1:77.27  ValLoss:2.0146  ValTop1:62.37  Best:64.74
Epoch 136/200  Time:43.6s  TrainLoss:1.4846  TrainTop1:77.87  ValLoss:1.9598  ValTop1:63.93  Best:64.74
Epoch 137/200  Time:43.7s  TrainLoss:1.4732  TrainTop1:78.32  ValLoss:2.0004  ValTop1:62.91  Best:64.74
Epoch 138/200  Time:44.0s  TrainLoss:1.4667  TrainTop1:78.56  ValLoss:2.0883  ValTop1:61.31  Best:64.74
Epoch 139/200  Time:43.5s  TrainLoss:1.4620  TrainTop1:78.57  ValLoss:1.9539  ValTop1:63.86  Best:64.74
Epoch 140/200  Time:44.1s  TrainLoss:1.4518  TrainTop1:78.92  ValLoss:1.9226  ValTop1:65.01  Best:65.01
Epoch 141/200  Time:43.7s  TrainLoss:1.4349  TrainTop1:79.47  ValLoss:1.9330  ValTop1:65.26  Best:65.26
Epoch 142/200  Time:44.3s  TrainLoss:1.4258  TrainTop1:79.80  ValLoss:1.9159  ValTop1:65.31  Best:65.31
Epoch 143/200  Time:44.1s  TrainLoss:1.4262  TrainTop1:79.82  ValLoss:1.9717  ValTop1:63.96  Best:65.31
Epoch 144/200  Time:44.4s  TrainLoss:1.4023  TrainTop1:80.49  ValLoss:2.0292  ValTop1:62.66  Best:65.31
Epoch 145/200  Time:43.6s  TrainLoss:1.4105  TrainTop1:80.55  ValLoss:1.9526  ValTop1:64.13  Best:65.31
Epoch 146/200  Time:44.7s  TrainLoss:1.3963  TrainTop1:80.86  ValLoss:1.8998  ValTop1:66.05  Best:66.05
Epoch 147/200  Time:43.6s  TrainLoss:1.3852  TrainTop1:81.45  ValLoss:1.9831  ValTop1:63.76  Best:66.05
Epoch 148/200  Time:44.8s  TrainLoss:1.3763  TrainTop1:81.47  ValLoss:1.8991  ValTop1:66.77  Best:66.77
Epoch 149/200  Time:44.2s  TrainLoss:1.3551  TrainTop1:82.51  ValLoss:1.9129  ValTop1:66.63  Best:66.77
Epoch 150/200  Time:44.6s  TrainLoss:1.3485  TrainTop1:82.59  ValLoss:1.9173  ValTop1:65.40  Best:66.77
Epoch 151/200  Time:43.7s  TrainLoss:1.3363  TrainTop1:83.00  ValLoss:1.8764  ValTop1:66.96  Best:66.96
Epoch 152/200  Time:44.4s  TrainLoss:1.3291  TrainTop1:83.44  ValLoss:1.9064  ValTop1:66.95  Best:66.96
Epoch 153/200  Time:44.0s  TrainLoss:1.3156  TrainTop1:83.68  ValLoss:1.9062  ValTop1:66.20  Best:66.96
Epoch 154/200  Time:44.0s  TrainLoss:1.3099  TrainTop1:83.74  ValLoss:1.8167  ValTop1:68.45  Best:68.45
Epoch 155/200  Time:44.1s  TrainLoss:1.2972  TrainTop1:84.18  ValLoss:1.9119  ValTop1:66.34  Best:68.45
Epoch 156/200  Time:43.8s  TrainLoss:1.2873  TrainTop1:84.58  ValLoss:1.9039  ValTop1:66.85  Best:68.45
Epoch 157/200  Time:44.0s  TrainLoss:1.2700  TrainTop1:85.28  ValLoss:1.8803  ValTop1:67.15  Best:68.45
Epoch 158/200  Time:43.9s  TrainLoss:1.2619  TrainTop1:85.57  ValLoss:1.8279  ValTop1:68.52  Best:68.52
Epoch 159/200  Time:44.4s  TrainLoss:1.2485  TrainTop1:85.95  ValLoss:1.9337  ValTop1:66.52  Best:68.52
Epoch 160/200  Time:43.9s  TrainLoss:1.2369  TrainTop1:86.52  ValLoss:1.8442  ValTop1:68.23  Best:68.52
Epoch 161/200  Time:44.5s  TrainLoss:1.2267  TrainTop1:86.98  ValLoss:1.8608  ValTop1:67.43  Best:68.52
Epoch 162/200  Time:43.8s  TrainLoss:1.2140  TrainTop1:87.16  ValLoss:1.8451  ValTop1:68.64  Best:68.64
Epoch 163/200  Time:44.5s  TrainLoss:1.1963  TrainTop1:87.91  ValLoss:1.8147  ValTop1:69.25  Best:69.25
Epoch 164/200  Time:43.9s  TrainLoss:1.1931  TrainTop1:88.00  ValLoss:1.8262  ValTop1:69.09  Best:69.25
Epoch 165/200  Time:44.6s  TrainLoss:1.1822  TrainTop1:88.52  ValLoss:1.8238  ValTop1:69.49  Best:69.49
Epoch 166/200  Time:43.9s  TrainLoss:1.1673  TrainTop1:89.08  ValLoss:1.8485  ValTop1:68.82  Best:69.49
Epoch 167/200  Time:44.5s  TrainLoss:1.1503  TrainTop1:89.66  ValLoss:1.8234  ValTop1:69.45  Best:69.49
Epoch 168/200  Time:43.6s  TrainLoss:1.1381  TrainTop1:90.03  ValLoss:1.8297  ValTop1:69.36  Best:69.49
Epoch 169/200  Time:44.3s  TrainLoss:1.1321  TrainTop1:90.18  ValLoss:1.8099  ValTop1:69.80  Best:69.80
Epoch 170/200  Time:44.0s  TrainLoss:1.1125  TrainTop1:91.02  ValLoss:1.8239  ValTop1:69.62  Best:69.80
Epoch 171/200  Time:44.7s  TrainLoss:1.1095  TrainTop1:91.06  ValLoss:1.8270  ValTop1:69.99  Best:69.99
Epoch 172/200  Time:43.9s  TrainLoss:1.0935  TrainTop1:91.73  ValLoss:1.8125  ValTop1:69.90  Best:69.99
Epoch 173/200  Time:44.6s  TrainLoss:1.0805  TrainTop1:92.29  ValLoss:1.8018  ValTop1:70.55  Best:70.55
Epoch 174/200  Time:44.1s  TrainLoss:1.0694  TrainTop1:92.59  ValLoss:1.8210  ValTop1:69.99  Best:70.55
Epoch 175/200  Time:45.0s  TrainLoss:1.0599  TrainTop1:93.06  ValLoss:1.7778  ValTop1:70.94  Best:70.94
Epoch 176/200  Time:44.0s  TrainLoss:1.0484  TrainTop1:93.49  ValLoss:1.7959  ValTop1:71.03  Best:71.03
Epoch 177/200  Time:44.7s  TrainLoss:1.0336  TrainTop1:93.96  ValLoss:1.7574  ValTop1:71.67  Best:71.67
Epoch 178/200  Time:43.9s  TrainLoss:1.0221  TrainTop1:94.49  ValLoss:1.7700  ValTop1:71.68  Best:71.68
Epoch 179/200  Time:44.9s  TrainLoss:1.0115  TrainTop1:94.71  ValLoss:1.7832  ValTop1:71.11  Best:71.68
Epoch 180/200  Time:44.0s  TrainLoss:1.0003  TrainTop1:95.23  ValLoss:1.7715  ValTop1:71.31  Best:71.68
Epoch 181/200  Time:45.0s  TrainLoss:0.9898  TrainTop1:95.63  ValLoss:1.7788  ValTop1:71.51  Best:71.68
Epoch 182/200  Time:43.8s  TrainLoss:0.9828  TrainTop1:95.84  ValLoss:1.7785  ValTop1:71.73  Best:71.73
Epoch 183/200  Time:44.7s  TrainLoss:0.9765  TrainTop1:96.04  ValLoss:1.7662  ValTop1:71.90  Best:71.90
Epoch 184/200  Time:43.9s  TrainLoss:0.9629  TrainTop1:96.60  ValLoss:1.7720  ValTop1:71.86  Best:71.90
Epoch 185/200  Time:44.3s  TrainLoss:0.9563  TrainTop1:96.89  ValLoss:1.7583  ValTop1:72.11  Best:72.11
Epoch 186/200  Time:44.1s  TrainLoss:0.9490  TrainTop1:97.12  ValLoss:1.7470  ValTop1:72.18  Best:72.18
Epoch 187/200  Time:43.9s  TrainLoss:0.9444  TrainTop1:97.18  ValLoss:1.7387  ValTop1:72.41  Best:72.41
Epoch 188/200  Time:44.4s  TrainLoss:0.9375  TrainTop1:97.47  ValLoss:1.7456  ValTop1:72.53  Best:72.53
Epoch 189/200  Time:43.7s  TrainLoss:0.9329  TrainTop1:97.59  ValLoss:1.7393  ValTop1:72.84  Best:72.84
Epoch 190/200  Time:44.3s  TrainLoss:0.9292  TrainTop1:97.80  ValLoss:1.7382  ValTop1:72.51  Best:72.84
Epoch 191/200  Time:44.6s  TrainLoss:0.9216  TrainTop1:98.04  ValLoss:1.7434  ValTop1:72.80  Best:72.84
Epoch 192/200  Time:45.0s  TrainLoss:0.9195  TrainTop1:98.16  ValLoss:1.7444  ValTop1:72.83  Best:72.84
Epoch 193/200  Time:43.9s  TrainLoss:0.9192  TrainTop1:98.09  ValLoss:1.7452  ValTop1:72.77  Best:72.84
Epoch 194/200  Time:44.3s  TrainLoss:0.9128  TrainTop1:98.33  ValLoss:1.7406  ValTop1:72.96  Best:72.96
Epoch 195/200  Time:44.2s  TrainLoss:0.9121  TrainTop1:98.30  ValLoss:1.7364  ValTop1:72.94  Best:72.96
Epoch 196/200  Time:44.3s  TrainLoss:0.9080  TrainTop1:98.48  ValLoss:1.7328  ValTop1:73.29  Best:73.29
Epoch 197/200  Time:44.2s  TrainLoss:0.9069  TrainTop1:98.50  ValLoss:1.7362  ValTop1:73.14  Best:73.29
Epoch 198/200  Time:44.1s  TrainLoss:0.9042  TrainTop1:98.61  ValLoss:1.7356  ValTop1:73.19  Best:73.29
Epoch 199/200  Time:43.9s  TrainLoss:0.9035  TrainTop1:98.66  ValLoss:1.7386  ValTop1:73.14  Best:73.29
Epoch 200/200  Time:44.4s  TrainLoss:0.9046  TrainTop1:98.60  ValLoss:1.7319  ValTop1:73.11  Best:73.29

```
---
