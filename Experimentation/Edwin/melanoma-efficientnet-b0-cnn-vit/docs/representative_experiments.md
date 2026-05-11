## 1. Initial EfficientNet-B0 baselines and robust validation

These experiments show the transition from a strong early single-fold model to a more reliable multi-fold validation protocol. E9 was the best early single-fold result, reaching a Final AUPRC of 0.2096, which suggested that full fine-tuning with BCE could produce useful ranking performance. However, because this result came from one validation split, it was not sufficient as a final reference.

E16 became more important methodologically because it used 3-fold OOF validation, TTA, `pos_weight`, and checkpoint selection by Val AUPRC. Although its OOF AUPRC was lower than E9’s single-fold AUPRC, it gave a more trustworthy estimate of generalisation. This experiment became the official pre-transfer Phase 2 baseline.

Overall, this line established that single-fold results could look stronger, but robust OOF evaluation was necessary for fair comparison.

| # | Experiment | Main change | LR | Epochs | Best Val AUROC / per fold | Best Val AUPRC / per fold | Best epoch(s) | Final Acc @0.5 | Final Precision @0.5 | Final Recall @0.5 | Final Specificity @0.5 | Final F1 @0.5 | Final AUROC | Final AUPRC | OOF AUROC | OOF AUPRC | Status / Conclusion |
|---:|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | Improved full fine-tuning + BCE | Upgraded BCE evaluation | 1e-5 | 10 | 0.9027 | N/D | 10 | 0.8856 | 0.1069 | 0.7131 | 0.8888 | 0.1859 | N/D | 0.2096 | N/D | N/D | Best historical single-fold |
| 16 | 3-fold + OOF + TTA + best by AUPRC | BCE + `pos_weight`, checkpoint by AUPRC | 1e-5 | 10/fold | 0.8876 / 0.8879 / 0.9084 at best-AUPRC ckpt | 0.1738 / 0.1455 / 0.2237 | 7 / 7 / 10 | 0.8674 | 0.0935 | 0.7536 | 0.8694 | 0.1664 | N/D | N/D | 0.8986 | 0.1811 | Official pre-transfer Phase 2 base |

## 2. Transfer learning to Phase 2

This line tested whether domain-pretrained EfficientNet-B0 checkpoints could improve Phase 2 performance. E22 showed that transfer learning was useful: it improved OOF AUPRC to 0.1990, clearly above the pre-transfer robust baseline E16. This confirmed that source-domain pretraining contained relevant visual information for melanoma classification.

E23 extended the same transfer setup to 20 epochs. This improved OOF AUPRC further to 0.2085, but OOF AUROC dropped to 0.8566. This separation between AUPRC and AUROC was important, because it showed that transfer improved the precision–recall ranking but did not improve all ranking metrics consistently.

Overall, transfer learning was positive compared with E16, but it was later surpassed by the stronger image-only focal recipe.

| # | Experiment | Main change | LR | Epochs | Best Val AUROC / per fold | Best Val AUPRC / per fold | Best epoch(s) | Final Acc @0.5 | Final Precision @0.5 | Final Recall @0.5 | Final Specificity @0.5 | Final F1 @0.5 | Final AUROC | Final AUPRC | OOF AUROC | OOF AUPRC | Status / Conclusion |
|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 22 | Phase 2 transfer from E21 + BCE | Head nueva, same E16 protocol | 1e-5 | 10/fold | 0.8889 / 0.8860 / 0.8742 | 0.2285 / 0.2115 / 0.1924 | 10 / 6 / 10 | 0.8281 | 0.0738 | 0.7568 | 0.8295 | 0.1344 | — | — | 0.8797 | 0.1990 | Positive transfer |
| 23 | Phase 2 transfer from E21 + BCE longer | Same E22, 20 epochs | 1e-5 | 20/fold | 0.8631 / 0.8881 / 0.8566 | 0.2376 / 0.2289 / 0.1958 | 16 / 9 / 19 | 0.8868 | 0.0997 | 0.6524 | 0.8943 | 0.1730 | — | — | 0.8566 | 0.2085 | Better AUPRC, lower AUROC |


## 3. Image-only recipe improvements

This was the strongest research line in the project. E33 showed that stronger image augmentations were beneficial, increasing OOF AUPRC to 0.2180. This created a better foundation than the previous transfer-based results.

The major improvement came from standard focal loss. E37, using `alpha=0.75` and `gamma=2.0`, reached OOF AUPRC 0.2325 and produced the best Phase 2 F1 operating point, with F1 0.3009 at threshold 0.5. This showed that focal loss improved both ranking and operational behaviour.

E39 extended the same recipe to 30 epochs and achieved the best overall image-only result, with OOF AUPRC 0.2347. Since OOF AUPRC is the main decision metric, E39 became the official baseline to beat. E49 improved F1 to 0.2937 by freezing BatchNorm, but reduced OOF AUPRC to 0.2221. Therefore, it was useful operationally but not selected as the main model.

| # | Experiment | Main change | LR | Epochs | Best Val AUROC / per fold | Best Val AUPRC / per fold | Best epoch(s) | Final Acc @0.5 | Final Precision @0.5 | Final Recall @0.5 | Final Specificity @0.5 | Final F1 @0.5 | Final AUROC | Final AUPRC | OOF AUROC | OOF AUPRC | Status / Conclusion |
|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 33 | Stronger augmentations + BCE + pos_weight | Stronger train aug only | 1e-5 | 20/fold | 0.8899 / 0.9006 / 0.8848 | 0.2155 / 0.2074 / 0.2197 | 19 / 19 / 20 | 0.8489 | 0.0816 | 0.7380 | 0.8509 | 0.1469 | 0.8877 | 0.2125 | 0.8872 | 0.2180 | Positive recipe change |
| 37 | Standard focal `alpha=0.75`, `gamma=2.0` | Increase positive focal alpha | 1e-5 | 20/fold | 0.8825 / 0.8826 / 0.8827 | 0.2257 / 0.2276 / 0.2468 | 20 / 20 / 19 | 0.9731 | 0.2775 | 0.3288 | 0.9846 | 0.3009 | 0.8830 | 0.2319 | 0.8814 | 0.2325 | Strong positive |
| 39 | Focal `alpha=0.75`, `gamma=2.0`, 30 epochs | Extended focal budget | 1e-5 | 30/fold | 0.8766 / 0.8823 / 0.8899 | 0.2305 / 0.2233 / 0.2686 | 25 / 23 / 28 | 0.9737 | 0.2737 | 0.2962 | 0.9859 | 0.2845 | 0.8762 | 0.2337 | 0.8781 | 0.2347 | Best official image-only by OOF AUPRC |
| 49 | Focal + freeze BatchNorm | BN kept in eval mode | 1e-5 | 30/fold | 0.8927 / 0.8967 / 0.8916 | 0.2084 / 0.2226 / 0.2565 | 27 / 12 / 26 | 0.9747 | 0.2895 | 0.2979 | 0.9869 | 0.2937 | N/D | N/D | 0.8836 | 0.2221 | Negative for OOF AUPRC |

## 4. Sampling, MixUp, EMA and augmentation artefacts

This line tested whether additional imbalance handling or regularisation could improve the focal baseline. E44 replaced loss weighting with WeightedRandomSampler and BCE. It reached OOF AUPRC 0.2190, which was below E39, so sampling alone was not a better solution.

E46 was the best MixUp variant, reaching OOF AUPRC 0.2245. This was a reasonable result, but it still did not match the focal baseline. E48 tested EMA on top of the focal recipe, but OOF AUPRC dropped to 0.2193. This showed that stabilising weights did not translate into better precision–recall ranking.

E53 tested real hair overlays and was strongly negative, with OOF AUPRC 0.2008. This was useful evidence that artefact augmentation can harm the model if it changes the image distribution in an unrealistic or overly aggressive way.

Overall, none of these additions improved the main metric.

| # | Experiment | Main change | LR | Epochs | Best Val AUROC / per fold | Best Val AUPRC / per fold | Best epoch(s) | Final Acc @0.5 | Final Precision @0.5 | Final Recall @0.5 | Final Specificity @0.5 | Final F1 @0.5 | Final AUROC | Final AUPRC | OOF AUROC | OOF AUPRC | Status / Conclusion |
|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 44 | WRS + BCE without `pos_weight` | Sampling instead of loss weighting | 1e-5 | 30/fold | 0.8601 / 0.8888 / 0.8485 | 0.2103 / 0.2293 / 0.2393 | 20 / 9 / 29 | 0.9341 | 0.1377 | 0.5205 | 0.9415 | 0.2178 | 0.8517 | 0.2086 | 0.8450 | 0.2190 | Negative main line |
| 46 | Focal + MixUp `alpha=0.2` | Add MixUp | 1e-5 | 30/fold | 0.8754 / 0.8825 / 0.8880 | 0.2305 / 0.2238 / 0.2389 | 29 / 28 / 28 | 0.9767 | 0.3048 | 0.2500 | 0.9898 | 0.2747 | 0.8784 | 0.2258 | 0.8820 | 0.2245 | Mixed but negative |
| 48 | Focal + EMA | EMA decay 0.999 | 1e-5 | 30/fold | 0.8800 / 0.8828 / 0.8761 | 0.2371 / 0.2092 / 0.2170 | 27 / 26 / 30 | 0.9736 | 0.2734 | 0.2997 | 0.9857 | 0.2859 | 0.8789 | 0.2185 | 0.8777 | 0.2193 | Negative OOF AUPRC |
| 53 | Focal + real hair overlays | `augV5` | 1e-5 | 30/fold | 0.8780 / 0.8870 / 0.8896 | 0.2158 / 0.2032 / 0.1921 | 30 / 24 / 24 | 0.9759 | 0.2771 | 0.2277 | 0.9893 | 0.2500 | N/D | N/D | 0.8859 | 0.2008 | Strongly negative AUPRC |

## 5. Hybrid CNN+ViT and hybrid transfer

This line explored whether CNN+ViT hybrids could improve the EfficientNet-B0 baseline. E57 was the best Kaggle-only hybrid. Its residual/gated design with conservative gate initialisation reached OOF AUPRC 0.2307, which was close to E39 but still lower. This suggested that the Transformer branch could add useful information, but not enough to replace the image-only focal baseline.

E78 tested transfer from a source-trained 14×14 hybrid checkpoint without SupCon. Despite stronger source-domain representation learning, the Kaggle OOF AUPRC was only 0.1908. E80 was the best hybrid transfer result, improving to OOF AUPRC 0.2091 with weak SupCon and conservative gating. However, it was still far below E39.

The main conclusion is that hybrid modelling was promising in architecture design, especially with residual gating, but source-domain hybrid pretraining did not transfer effectively to Phase 2.

| # | Experiment | Main change | LR | Epochs | Best Val AUROC / per fold | Best Val AUPRC / per fold | Best epoch(s) | Final Acc @0.5 | Final Precision @0.5 | Final Recall @0.5 | Final Specificity @0.5 | Final F1 @0.5 | Final AUROC | Final AUPRC | OOF AUROC | OOF AUPRC | Status / Conclusion |
|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 57 | Residual/gated hybrid v1 | Conservative gate init `-3.0` | CNN 1e-5 / new 5e-5 | 15/fold | 0.8716 / 0.8752 / 0.8803 | 0.2272 / 0.2176 / 0.2638 | 2 / 2 / 2 | 0.9688 | 0.2344 | 0.3408 | 0.9800 | 0.2777 | N/D | N/D | 0.8746 | 0.2307 | Best Kaggle-only hybrid, below E39 |
| 78 | Kaggle transfer hybrid 14×14 no-head no-SupCon | Source checkpoint transfer, no SupCon | CNN 1e-5→5e-6 / new 1e-3→1e-5 | 30 | 0.8875 / 0.8848 / 0.8827 | 0.2145 / 0.1890 / 0.1869 | 29 / 22 / 29 | 0.9732 | 0.2651 | 0.2928 | 0.9854 | 0.2783 | N/D | N/D | 0.8827 | 0.1908 | Negative vs E39 |
| 80 | Kaggle transfer hybrid + weak SupCon | Gate -3.0, SupCon 0.01, alpha 0.8 | CNN 1e-5→1e-6 / new 1e-4→1e-5 | 30 ES: 24 / 17 / 19 | 0.8971 / 0.8889 / 0.8987 | 0.2290 / 0.1960 / 0.2102 | 18 / 11 / 13 | 0.8682 | 0.0924 | 0.7329 | 0.8708 | 0.1641 | N/D | N/D | 0.8938 | 0.2091 | Best transfer E78–E83, still negative vs E39 |

