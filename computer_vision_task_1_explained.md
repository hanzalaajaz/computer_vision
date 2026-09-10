# Skin Lesion Classification — Results & Explanation

## Run Setup

- **Dataset:** Kaggle `nodoubttome/skin-cancer9-classesisic` ("Skin Cancer 9 Classes ISIC")
- **Classes used (5 of 9):** basal cell carcinoma, dermatofibroma, melanoma, nevus, vascular lesion
- **Split:** Train 1265 images / Val 140 / Test 67 (stratified 90/10 split of Train, Test held out separately)
- **Training setup:** 10 epochs, batch size 16, Adam optimizer (lr 1e-4), ImageNet-pretrained backbones fine-tuned end-to-end
- **Device:** GPU (CUDA)

## Table 1 — Transfer Learning Model Comparison

Eight ImageNet-pretrained CNN backbones were fine-tuned end-to-end on the 5-class dataset and evaluated on the held-out test set.

| Model | Acc% | Prec% | Rec% | F1% | AUC% |
|---|---|---|---|---|---|
| AlexNet | 73.13 | 78.15 | 73.13 | 72.19 | 89.14 |
| VGG16 | 59.7 | 56.87 | 59.7 | 53.1 | 84.66 |
| VGG19 | 64.18 | 67.27 | 64.18 | 61.08 | 88.65 |
| ResNet18 | 65.67 | 66.64 | 65.67 | 61.05 | 86.42 |
| ResNet50 | 71.64 | 79.78 | 71.64 | 68.96 | 91.62 |
| ResNet101 | 70.15 | 76.91 | 70.15 | 67.09 | 91.02 |
| DenseNet121 | 70.15 | 81.07 | 70.15 | 68.37 | 92.65 |
| EfficientNet-B0 | 56.72 | 56.98 | 56.72 | 51.2 | 86.98 |

**Reading the results:** AlexNet came out on top by raw accuracy (73.13%), with ResNet50 close behind (71.64%). This is a bit unusual next to the reference paper, where the deeper/more modern backbones (ResNet50, DenseNet, EfficientNet) generally beat AlexNet — with only 10 epochs and a small dataset (1265 training images across 5 classes), the smaller AlexNet has fewer parameters to overfit with, while the deeper nets (VGG16/19, EfficientNet-B0) likely needed more epochs or a lower learning rate to converge properly here. Notice DenseNet121 and ResNet50/101 have the **highest AUC scores** (91–93%) despite lower raw accuracy — meaning their class-separation ability is actually strong, they're just less well-calibrated at the default decision threshold within only 10 epochs of fine-tuning.

## Table 2 — Deep Features + Classical Classifiers

AlexNet (the best Table 1 backbone) was frozen and used as a fixed feature extractor; seven classical ML classifiers were then trained on those deep features.

| Classifier | Acc% | Prec% | Rec% | F1% | AUC% |
|---|---|---|---|---|---|
| Logistic Regression | 65.67 | 69.84 | 65.67 | 62.71 | 83.76 |
| Decision Tree | 67.16 | 68.46 | 67.16 | 65.28 | 78.43 |
| Random Forest | 67.16 | 71.91 | 67.16 | 64.37 | 88.69 |
| K-Nearest Neighbors | 70.15 | 74.02 | 70.15 | 68.42 | 85.65 |
| Linear SVM | 64.18 | 70.14 | 64.18 | 61.32 | 83.76 |
| RBF-SVM | 62.69 | 67.41 | 62.69 | 59.05 | 84.99 |
| XGBoost | 64.18 | 67.83 | 64.18 | 60.51 | 84.34 |

**Reading the results:** K-Nearest Neighbors gave the best accuracy (70.15%) of the classical classifiers, edging out Random Forest and Decision Tree (both 67.16%). None of the classical classifiers beat AlexNet's own end-to-end fine-tuned accuracy (73.13%) — expected, since fine-tuning updates the whole network for this specific task, while these classifiers only ever see frozen, generic AlexNet features. Random Forest had the best AUC (88.69%) among the classical models, suggesting it separates classes more reliably even though KNN edges it slightly on raw accuracy.

## Table 3 — Computational Efficiency Comparison

Parameter count, on-disk model size, and single-image inference latency for each backbone, alongside its Table 1 accuracy.

| Model | Params(M) | Size(MB) | FLOPs(G) | Infer(ms) | Acc% |
|---|---|---|---|---|---|
| AlexNet | 57.02 | 217.53 | N/A | 2.1 | 73.13 |
| VGG16 | 134.28 | 512.24 | N/A | 9.58 | 59.7 |
| VGG19 | 139.59 | 532.5 | N/A | 11.73 | 64.18 |
| ResNet18 | 11.18 | 42.64 | N/A | 3.34 | 65.67 |
| ResNet50 | 23.52 | 89.72 | N/A | 5.89 | 71.64 |
| ResNet101 | 42.51 | 162.16 | N/A | 11.73 | 70.15 |
| DenseNet121 | 6.96 | 26.55 | N/A | 16.66 | 70.15 |
| EfficientNet-B0 | 4.01 | 15.31 | N/A | 9.02 | 56.72 |

**Reading the results:** AlexNet is both the most accurate *and* the fastest to run (2.1ms), which is part of why it "won" Table 1 — with only 10 epochs of training, simpler/faster-converging architectures had an advantage. ResNet50 is the standout efficiency pick if you value the accuracy/size trade-off: it's 71.64% accurate at less than half AlexNet's parameter count (23.5M vs 57M) and under a third its disk size. DenseNet121 and EfficientNet-B0 are by far the smallest models (7M and 4M parameters) but pay for it with the slowest inference times here (16.7ms and 9.0ms) — likely because their layer structure (dense connections / depthwise separable convolutions) isn't as well-optimized for small-batch GPU inference as it is for parameter efficiency.

**Note:** the FLOPs (G) column reads `N/A` for every model — the `thop` profiling step failed to run during this session, so those values are still missing. Everything else in the table is genuine measured output.

## Overall Takeaway

With only 10 training epochs on a relatively small, 5-class subset of the dataset, the simpler/shallower architectures (AlexNet, ResNet50) outperformed the deeper ones (VGG16/19, EfficientNet-B0) on raw accuracy — the opposite of what you'd expect with longer training. The AUC column is the more trustworthy signal of each model's actual discriminative ability here: ResNet50, ResNet101, and DenseNet121 all score 91%+ AUC despite middling accuracy, meaning more training epochs would likely let them close the gap with (or beat) AlexNet on accuracy too.
