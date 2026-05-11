# SIIM-ISIC Melanoma Classification
A Kaggle melanoma classification challenge on the SIIM-ISIC dataset.
https://www.kaggle.com/competitions/siim-isic-melanoma-classification/

## Model Performance
| Notebook (.ipynb)          | Model                                                         | Description                                         | Val AUC | Kaggle Test AUC |
|----------------------------|---------------------------------------------------------------|-----------------------------------------------------|---------|-----------------|
| train_b0-hybrid_E80_edwin  | Hybrid EfficientNet-B0 + Transformer (Image only)             | Best Kaggle hybrid from Edwin                       | 0.8938   | 0.8834          |
| train_b7_omar              | EfficientNet-B7 (Metadata + Image)                            | Based on Louis' CW-5 notebook with numerous changes | 0.9210  | 0.8818          |
| create_mean_blend_ensemble | Mean probability blend of the above models (Metadata + Image) | Our best setup so far, 49/7 TP/FN                   | __0.9766*__ | __0.8992__          |

\* Due to multiple model authors with different split creation, and Kaggle holding out the test set, we believe some training data was leaked through validation. Future efforts should adapt create_mean_blend_ensemble.ipynb to only use validation data held-out from models.
