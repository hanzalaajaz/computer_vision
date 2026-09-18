# Lab Task 02 — Effect of Image Filtering on Skin-Lesion Classification

**Name:** Hanzala Ajaz
**Registration No:** FA23-BAI-016

---

## 1. Dataset & Setup

- **Dataset:** HAM10000 (Kaggle `kmader/skin-cancer-mnist-ham10000`) — the same dataset used by the Task 01 reference paper.
- **Classes used (5 of 7):** basal cell carcinoma, dermatofibroma, melanoma, melanocytic nevi, vascular lesions

| Class | Count |
|---|---|
| Melanocytic nevi (nv) | 6705 |
| Melanoma (mel) | 1113 |
| Basal cell carcinoma (bcc) | 514 |
| Vascular lesions (vasc) | 142 |
| Dermatofibroma (df) | 115 |

- **Split:** Train 6871 / Val 859 / Test 859 (stratified 80/10/10)
- **Models used (best 3 from Task 01):** AlexNet, ResNet50, DenseNet121
- **Filters tested:** No Filter (baseline), Average, Gaussian, Median, Sharpening, Sobel
- **Training:** 6 epochs, batch size 16, Adam optimizer (lr 1e-4), fine-tuned end-to-end
- **Device:** GPU (CUDA)

18 experiments were run in total (3 models × 6 filter conditions).

---

## 2. Results Table

| Model | Filter | Accuracy | Precision | Recall | F1-score | Macro-F1 | Balanced Accuracy | AUC |
|---|---|---|---|---|---|---|---|---|
| AlexNet | No Filter | 87.89 | 87.35 | 87.89 | 87.15 | 72.78 | 72.73 | 96.77 |
| ResNet50 | No Filter | 91.39 | 90.99 | 91.39 | 90.66 | 79.77 | 75.38 | 97.18 |
| DenseNet121 | No Filter | 90.69 | 90.63 | 90.69 | 90.51 | 79.58 | 82.10 | 97.63 |
| AlexNet | Average | 88.82 | 88.20 | 88.82 | 88.11 | 72.60 | 73.03 | 97.17 |
| AlexNet | Gaussian | 88.13 | 87.31 | 88.13 | 86.57 | 74.21 | 72.17 | 97.19 |
| AlexNet | Median | 88.59 | 87.58 | 88.59 | 87.94 | 62.88 | 68.32 | 97.06 |
| AlexNet | Sharpening | 88.36 | 87.85 | 88.36 | 87.96 | 73.45 | 71.31 | 96.75 |
| AlexNet | Sobel | 81.14 | 78.29 | 81.14 | 76.94 | 39.26 | 34.72 | 86.18 |
| ResNet50 | Average | 89.41 | 89.67 | 89.41 | 89.37 | 80.05 | 80.95 | 97.40 |
| ResNet50 | Gaussian | 90.57 | 89.88 | 90.57 | 89.66 | 83.48 | 79.99 | 97.62 |
| ResNet50 | Median | 91.27 | 90.83 | 91.27 | 90.52 | 79.90 | 77.27 | 97.48 |
| ResNet50 | Sharpening | 91.97 | 91.73 | 91.97 | 91.60 | 82.27 | 82.52 | 97.26 |
| ResNet50 | Sobel | 84.52 | 82.86 | 84.52 | 82.64 | 57.94 | 52.55 | 91.88 |
| DenseNet121 | Average | 88.24 | 87.51 | 88.24 | 87.00 | 71.60 | 71.99 | 96.86 |
| DenseNet121 | Gaussian | 92.32 | 92.07 | 92.32 | 91.94 | 83.00 | 84.25 | 97.78 |
| DenseNet121 | Median | 91.27 | 90.94 | 91.27 | 90.91 | 78.85 | 75.08 | 96.70 |
| DenseNet121 | Sharpening | 91.39 | 91.02 | 91.39 | 91.03 | 83.10 | 79.52 | 98.41 |
| DenseNet121 | Sobel | 83.70 | 83.04 | 83.70 | 83.21 | 56.06 | 52.40 | 91.98 |

**Best overall result:** DenseNet121 + Gaussian filter — 92.32% accuracy, 97.78% AUC.
**Worst overall result:** AlexNet + Sobel filter — 81.14% accuracy, 86.18% AUC.

---

## 3. Comparative Analysis

### Accuracy change vs. baseline (No Filter), percentage points

| Filter | AlexNet | DenseNet121 | ResNet50 |
|---|---|---|---|
| No Filter | 0 | 0 | 0 |
| Average | +0.93 | -2.45 | -1.98 |
| Gaussian | +0.24 | +1.63 | -0.82 |
| Median | +0.70 | +0.58 | -0.12 |
| Sharpening | +0.47 | +0.70 | +0.58 |
| Sobel | -6.75 | -6.99 | -6.87 |

### Which filter caused the greatest change? (avg. absolute accuracy delta across models)

| Filter | Avg |Δ| |
|---|---|
| Sobel | 6.87 |
| Average | 1.79 |
| Gaussian | 0.90 |
| Sharpening | 0.58 |
| Median | 0.47 |

**Sobel produced by far the largest average shift in accuracy** across all three models — and always in the negative direction.

### Is each filter's effect consistent across models?

- **Average:** NOT consistent (AlexNet +, DenseNet121 −, ResNet50 −)
- **Gaussian:** NOT consistent (AlexNet +, DenseNet121 +, ResNet50 −)
- **Median:** NOT consistent (AlexNet +, DenseNet121 +, ResNet50 −)
- **Sharpening:** Consistent — improved accuracy for all 3 models
- **Sobel:** Consistent — hurt accuracy for all 3 models

### Macro-F1 change vs. baseline

| Filter | AlexNet | DenseNet121 | ResNet50 |
|---|---|---|---|
| Average | -0.18 | -7.98 | +0.28 |
| Gaussian | +1.43 | +3.42 | +3.71 |
| Median | -9.90 | -0.73 | +0.13 |
| Sharpening | +0.67 | +3.52 | +2.50 |
| Sobel | -33.52 | -23.52 | -21.83 |

### Balanced Accuracy change vs. baseline

| Filter | AlexNet | DenseNet121 | ResNet50 |
|---|---|---|---|
| Average | +0.30 | -10.11 | +5.57 |
| Gaussian | -0.56 | +2.15 | +4.61 |
| Median | -4.41 | -7.02 | +1.89 |
| Sharpening | -1.42 | -2.58 | +7.14 |
| Sobel | -38.01 | -29.70 | -22.83 |

Sobel's damage is far more severe on Macro-F1 and Balanced Accuracy than on raw Accuracy — a sign that it disproportionately wrecks performance on the minority classes (dermatofibroma, vascular lesions) while raw accuracy stays propped up by the dominant "melanocytic nevi" class.

---

## 4. Questions to Answer

**1. Which three pretrained models performed best in Lab Activity 1?**
AlexNet, ResNet50, DenseNet121.

**2. How does filtering affect each of the three models?**
ResNet50 and DenseNet121 both tolerate — and sometimes benefit from — Gaussian, Median, and Sharpening filters, while AlexNet is more mixed (small gains from most filters but the sharpest Sobel-induced collapse in Macro-F1, -33.52 points). All three models are severely degraded by Sobel.

**3. Which filter produces the greatest change compared with the unfiltered baseline?**
**Sobel**, by a wide margin (avg. 6.87-point accuracy drop, and 20-30+ point drops in Macro-F1/Balanced Accuracy).

**4. Does the effect of a filter remain consistent across all three models?**
Only **Sharpening** (consistently positive) and **Sobel** (consistently negative) have the same direction of effect across all 3 models. Average, Gaussian, and Median all disagree in direction between models.

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**
On average, filtering **decreases** macro-F1 relative to baseline — driven almost entirely by Sobel's severe drop. Excluding Sobel, Gaussian and Sharpening actually *improve* macro-F1 for all three models.

**6. Which lesion classes are most affected by filtering?**
The minority classes (dermatofibroma: 115 images, vascular lesions: 142 images) are most filter-sensitive — this is visible indirectly through the large gap between the Accuracy and Macro-F1/Balanced-Accuracy drops under Sobel: raw accuracy is propped up by the dominant "melanocytic nevi" class (6705 images) even as minority-class performance collapses.

**7. Why might smoothing remove useful lesion texture or morphological information?**
Average, Gaussian, and Median filters all work by locally averaging (or taking the local median of) neighboring pixel intensities. Fine lesion details — pigment networks, dots, streaks — live in the image's high-frequency content, which smoothing directly attenuates. Coarse shape and color survive, but the fine texture cues that separate visually similar classes can be blurred away.

**8. Why might sharpening or edge detection help or hurt classification?**
Sharpening amplifies high-frequency detail, which can make lesion borders and texture more visible — helping in this run (consistently positive across all 3 models). But it can also amplify noise and artifacts. Sobel edge detection goes further and discards almost all color and intensity information, keeping only edge/gradient magnitude — which, based on this run's results, hurt every model badly, since color and shading are clearly important cues for this classification task that pure edge maps throw away entirely.

**9. What is the difference between convolution and correlation?**
Both slide a kernel over an image and compute a weighted sum at each position. True convolution flips the kernel both horizontally and vertically before sliding it; correlation does not. For a symmetric kernel (e.g. Gaussian) the two are identical; for an asymmetric kernel (e.g. Sobel) they produce mirrored results. In practice, deep learning frameworks' "Conv2d" layers actually implement cross-correlation, not true convolution.

**10. What is the relationship between classical image processing and deep-learning-based feature extraction?**
Classical filters are fixed, hand-designed kernels that extract one specific type of feature (a blur, an edge map) uniformly across every image. A CNN's convolutional layers perform the same fundamental sliding-kernel operation, but the kernel weights are *learned* from data, and a deep network learns many of them simultaneously, optimized specifically for the classification task. Applying a classical filter before a CNN, as in this experiment, effectively hand-picks one extra fixed feature-extraction stage in front of a system that already learns its own — which is exactly why the effect is filter- and model-dependent: Sobel destroys information (color, shading) the CNN would otherwise have learned to use, while Sharpening pre-emphasizes genuinely useful edge information without discarding anything, which is why it helped instead.

---

## 5. Overall Conclusion

DenseNet121 with a Gaussian filter gave the best overall result (92.32% accuracy, 97.78% AUC), narrowly ahead of its own unfiltered baseline and ResNet50's best filtered result. Sharpening was the only filter that reliably helped all three models, making it the safest general-purpose preprocessing choice among those tested. Sobel should be avoided for this task — while its raw accuracy drop (~7 points) looks moderate, its damage to Macro-F1 and Balanced Accuracy (20–38 points) shows it severely harms performance on the minority lesion classes, which matters most for a medical classification task where rare, dangerous classes (dermatofibroma, vascular lesions) must not be missed.
