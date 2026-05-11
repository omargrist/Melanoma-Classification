# Melanoma Classification with EfficientNet-B0 and CNN+ViT

This repository contains a binary melanoma classification project for melanoma detection using PyTorch, EfficientNet-B0 and CNN+ViT hybrid models.

The project is based on the SIIM-ISIC / ISIC 2020 melanoma classification dataset, using resized 224×224 JPEG images. The main objective was to build and compare robust image-only models under a patient-level grouped validation protocol.

## Main objective

The primary goal was to improve melanoma classification performance under strong class imbalance.

The main model-selection metric was:

**Out-of-fold Average Precision (OOF AUPRC)**

AUROC, precision, recall, specificity, F1 score and threshold sweeps were used as secondary and operational metrics, but not as the main selection criterion.

## Experimental protocol

The main experiments used:

- EfficientNet-B0 as the reference backbone
- PyTorch training pipeline
- 224×224 resized JPEG images
- Patient-level grouped splits with StratifiedGroupKFold
- 3-fold out-of-fold validation
- Test-time augmentation when applicable
- Binary output with a single final logit
- Checkpoint selection mainly by validation AUPRC

## Main experimental lines

This repository focuses on four main research lines:

1. Initial EfficientNet-B0 baselines and robust validation
2. Transfer learning to Phase 2
3. Image-only recipe improvements with stronger augmentation and focal loss
4. CNN+ViT hybrid models and hybrid transfer

Metadata experiments were explored during development but are not included in this repository, as they did not improve the main image-only direction.

## Main result

The best official model was Experiment E39.

| Experiment | Model | Main recipe | OOF AUROC | OOF AUPRC |
|---|---|---|---:|---:|
| E39 | EfficientNet-B0 | Strong augmentations + focal loss `alpha=0.75`, `gamma=2.0`, 30 epochs | 0.8781 | 0.2347 |

E39 is the official baseline to beat because it achieved the highest OOF AUPRC.

## Most representative experiments

| Experiment | Line | Role | OOF AUROC | OOF AUPRC |
|---|---|---|---:|---:|
| E16 | EfficientNet-B0 baseline | First robust pre-transfer baseline | 0.8986 | 0.1811 |
| E23 | Transfer learning | Best EfficientNet-B0 transfer result | 0.8566 | 0.2085 |
| E39 | Image-only recipe | Best official image-only baseline | 0.8781 | 0.2347 |
| E57 | CNN+ViT hybrid | Best Kaggle-only hybrid | 0.8746 | 0.2307 |
| E80 | Hybrid transfer | Best hybrid transfer result | 0.8938 | 0.2091 |

## Uploaded notebooks

The repository includes selected notebooks only, rather than every experiment.

```text
notebooks/
  01_E23_melanoma_classifier_transfer_full_ft_BCE_3fold_TTA_AUPRC_20epochs_pos_weight.ipynb
  02_E39_melanoma_classifier_transfer_full_ft_FOCAL_3fold_TTA_AUPRC_30epochs_aug.ipynb
  03_E57_melanoma_classifier_custom_transferKaggleCNN_ViT_residualGated_hybrid_fullCNN_3folds_OOF_AUPRC_v1.ipynb