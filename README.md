# Blood Cell Classification - Deep Learning Project

CNN, ResNet & Vision Transformer experiments on BloodMNIST dataset.

## 🩸 Dataset
- **BloodMNIST** (8 blood cell types)
- 28×28 RGB images
- 11,959 train / 1,712 validation / 3,421 test images
- Classes: Basophil, Eosinophil, Erythroblast, IG, Lymphocyte, Monocyte, Neutrophil, Platelet

## 🏗️ Models Implemented

### 1. Custom CNN Architectures
- Base CNN: Conv → ReLU → MaxPool blocks
- With BatchNorm / LayerNorm
- With Dropout (0.05, 0.2, 0.5, 0.7)
- With Weight Decay (1e-2 to 1e-5)
- Optimizers: Adam, SGD, RMSprop, Nesterov Adam
- Learning rate scheduling

### 2. Transfer Learning
- **ResNet18** (torchvision)
- Feature extraction (frozen backbone)
- Fine-tuning (last 2 blocks unfrozen)

### 3. Vision Transformer
- Pre-trained ViT (DeiT-tiny via `timm`)
- Input: 224×224 (resized from 28×28)
- Feature extraction & fine-tuning

### 4. Ensemble
- **Soft Voting** of 7 best models
- **Highest accuracy achieved**

## 📊 Results Summary

| Model | Best Accuracy | Notes |
|-------|--------------|-------|
| CNN (BatchNorm + Adam) | 94.68% | Good balance |
| CNN with Dropout (p=0.05) | 94.33% | Reduced overfitting |
| ResNet18 (fine-tuned) | 87.40% | Underperformed custom CNN |
| Vision Transformer (fine-tuned) | 90.18% | Better than ResNet |
| **Ensemble (Soft Voting)** | **96.55%** | **Best result** |

## 🔧 Tech Stack
- Python 3.8+
- PyTorch, torchvision
- timm (for ViT)
- numpy, matplotlib, scikit-learn
