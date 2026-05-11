# Melanoma Classification

This folder contains my personal contribution to the group melanoma classification project. The notebooks document an iterative development pipeline, with each notebook building on the previous by adding one key feature or experiment. All notebooks run on Kaggle with the SIIM-ISIC Melanoma Classification dataset.

---

## Notebook Pipeline

### NB1, `melanoma-cw.ipynb`
**Purpose:** Base EfficientNet-B3 training pipeline with no metadata.

**What it does:**
- Loads JPEG images directly from Kaggle (`/kaggle/input/`)
- Stratified 80/20 train/val split preserving class ratio
- Augmentation: horizontal/vertical flip, rotation, colour jitter, affine, random erasing
- EfficientNet-B3 backbone via `timm` with frozen backbone and custom dropout head
- Two-phase training: backbone frozen, unfreezes at epoch 3
- Class-weighted CrossEntropyLoss to handle 98/2 imbalance so the model doesn't just predict benign every time
- Saves best model checkpoint by AUC
- Evaluation: confusion matrix, loss curves, AUC curve

**Key settings:** Image size 224x224 | Batch size 64 | 10 epochs | LR 1e-4

---

### NB2, `melanoma-cw-2-metadata.ipynb`
**Purpose:** Adds patient metadata preprocessing and a separate metadata MLP model.

**What it adds over NB1:**
- Metadata preprocessing: sex (binary), normalised age, one-hot anatomical site (9 features total)
- Separate MetadataModel, two-layer MLP (64, 32 hidden units)
- Metadata model trained independently for 30 epochs
- Loads pre-trained image model checkpoint from Kaggle outputs
- Ensemble: 80% image model + 20% metadata model
- Ensemble results table with AUC for each component

**Key settings:** Same image pipeline as NB1 | Metadata model: AdamW lr=1e-3 | 30 metadata epochs

---

### NB3, `melanoma-cw-3-combined.ipynb`
**Purpose:** Introduces a joint multimodal model that fuses image and metadata features internally.

**What it adds over NB2:**
- MelanomaModel now has two branches, EfficientNet-B3 image branch and metadata MLP branch, concatenated before the classification head
- Dataset class updated to return (image, meta, label) triplets
- ROC curve plotted for image model, metadata model and ensemble
- Grad-CAM visualisation added, hooks into backbone.blocks[-1] to highlight attended regions
- Side-by-side original image and heatmap overlay for sample validation images

**Key settings:** Image size 224x224 | Batch size 64 | meta_dim=9

---

### NB4, `melanoma-cw-4-TTA.ipynb`
**Purpose:** Adds Test-Time Augmentation for improved inference.

**What it adds over NB3:**
- TTA over 8 geometric transforms: identity, horizontal flip, vertical flip, 180-degree rotation, and their transpositions
- Softmax probabilities averaged across all 8 views before ensemble blending
- Loads separate image and metadata checkpoints rather than the joint model
- Updated metadata features include n_images (log-scaled per-patient count) and image_size (log file size), 11 features total
- Full evaluation: confusion matrix, classification report, ROC curve, Grad-CAM

**Key settings:** N_TTA=8 | Image size 224x224 | 11 metadata features

---

### NB5, `melanoma-cw-5-retrain.ipynb`
**Purpose:** Full retrain with larger images, mixed precision and extended training.

**What it adds over NB4:**
- Image size increased to 320x320
- Mixed precision training using torch.cuda.amp.autocast and GradScaler for faster GPU training
- UNFREEZE_AT=4, backbone frozen for 4 epochs before full fine-tuning
- Differential learning rates on unfreeze: backbone 1e-5, head 1e-4
- 15 epochs total
- Stronger colour jitter (brightness/contrast 0.3 vs 0.2)
- Batch size reduced to 48 to fit larger images in GPU memory

**Key results:**
| Model | AUC-ROC |
|---|---|
| EfficientNet-B3 (no TTA) | 0.8656 |
| EfficientNet-B3 (TTA x8) | 0.8673 |
| Metadata MLP | 0.7621 |
| Ensemble (80/20) | 0.8700 |

**Key settings:** Image size 320x320 | Batch size 48 | 15 epochs | Mixed precision

**This is the last version using efficientnet-B3**

---

### NB6, `melanoma-cw-6-efficientnetB4.ipynb`
**Purpose:** Architecture comparison, EfficientNet-B4 in place of B3.

**What changes from NB5:**
- Backbone changed to efficientnet_b4 for comparison
- Batch size reduced to 16 due to increased GPU memory requirements of B4
- Gradient clipping added (max_norm=1.0) to prevent loss explosion
- LR reduced to 1e-4 (B4 more sensitive to high LR)
- autocast removed from validation loop to prevent NaN probabilities
- Saves to best_model_efficientnet_b4.pth, separate checkpoint from B3

**Notes:** B4 is approximately 50% slower per epoch than B3 at the same image size due to the larger architecture. Used as an architecture comparison experiment rather than a replacement for the B3 results.

**Key settings:** Image size 320x320 | Batch size 16 | 15 epochs | efficientnet_b4

---

### NB7, `melanoma-cw-7-efficientnetB1.ipynb`
**Purpose:** Architecture comparison, EfficientNet-B1 as a smaller faster alternative.

**What changes from NB5:**
- Backbone changed to efficientnet_b1 for comparison
- Image size reduced to 240x240 (B1 native resolution, faster and less memory)
- Batch size increased back to 48 (smaller images allow this)
- 10 epochs set to fit within Kaggle 12-hour GPU limit
- Saves to best_model_efficientnet_b1.pth

**Key result:** Val AUC 0.8554 at epoch 16 before timeout, competitive with B3 (0.8700) despite significantly fewer parameters, demonstrating a strong efficiency-accuracy tradeoff.

**Key settings:** Image size 240x240 | Batch size 48 | 10 epochs | efficientnet_b1

---

## Summary Table

| Notebook | Backbone | Image Size | Key Addition | Best AUC |
|---|---|---|---|---|
| NB1 | EfficientNet-B3 | 224 | Base pipeline | 0.8056 |
| NB2 | EfficientNet-B3 | 224 | Metadata model + ensemble | 0.8153 |
| NB3 | EfficientNet-B3 | 224 | Joint model + GradCAM + ROC | 0.8149 |
| NB4 | EfficientNet-B3 | 224 | TTA (x8) + 11 metadata features | 0.8377 |
| NB5 | EfficientNet-B3 | 320 | Mixed precision + full retrain | 0.8700 |
| NB6 | EfficientNet-B4 | 320 | B4 architecture comparison | 0.8664 |
| NB7 | EfficientNet-B1 | 240 | B1 efficiency comparison | 0.8554 |

---

## Environment

- **Platform:** Kaggle Notebooks (T4 GPU, 16GB VRAM)
- **Framework:** PyTorch
- **Key libraries:** timm, torchvision, sklearn, pandas, matplotlib
---

## Key Findings

1. **Oversampling causes overfitting**, repeating 584 minority images caused the model to memorise training samples. Class-weighted loss is more effective.
2. **Two-phase freezing is essential**, training the full model from epoch 1 with a randomly initialised head produces unstable gradients. Freezing the backbone for warmup stabilises early training.
3. **Metadata adds value**, patient demographics provide a small but consistent AUC improvement when blended at 20% weight.
4. **TTA improves AUC**, averaging predictions over 8 geometric transforms gives a free performance boost with no additional training.
5. **B1 is surprisingly competitive**, EfficientNet-B1 at 240px reached 0.8554 AUC with significantly fewer parameters and faster training than B3, making it a strong choice when GPU time is limited.
6. **Threshold matters clinically**, lowering the classification threshold from 0.5 to 0.3 reduced missed melanoma cases from 47 to 25, demonstrating the sensitivity-specificity tradeoff relevant to clinical deployment.
