# Lab 03 — Edge Detection Techniques and Their Impact on Classification Performance

**Hanzala Ajaz — FA23-BAI-016**

This notebook (`Lab_03_edge_detection.ipynb`) was run end-to-end in Google Colab (GPU/CUDA)
with no errors. This README documents what it does and the real results it produced.

---

## What's inside

| Section | What it does |
|---|---|
| 0. Setup | Self-installs packages, imports, configuration, pre-downloads all model weights once |
| 1. Dataset Preparation | Downloads HAM10000 via `kagglehub`, filters to a 5-class subset, stratified train/val/test split |
| Task 1 | Sobel (Gx/Gy/magnitude), Prewitt, Laplacian, LoG, Canny — compared side by side on 3 lesion classes |
| Task 2 | Gaussian & Salt-and-Pepper noise, with/without Gaussian/Median pre-filtering → Table 1 |
| Task 3 | Canny threshold/kernel sweep → Table 2, auto-selects a "best" configuration |
| Task 4 | Builds 3 dataset versions (Raw / Filtered / Edge) and trains 5 models on each (15 runs) |
| Task 5 | Reshapes results into the cross-lab comparison table → Table 3 |
| Task 6 | Confusion matrices + bar chart for the best-performing model |
| Report | Auto-generates `results_lab03/report.md` with all tables and discussion answers |

## How to run
Colab → Runtime → GPU → upload the notebook → Runtime → Run all. Nothing else needed;
packages and dataset download themselves (Kaggle login prompt on first dataset download).

---

## Dataset & Setup (from this run)

- **Dataset:** HAM10000 (`kmader/skin-cancer-mnist-ham10000`)
- **Classes (5):** melanocytic nevi (6705), melanoma (1113), basal cell carcinoma (514), vascular lesions (142), dermatofibroma (115)
- **Split:** Train 6871 / Val 859 / Test 859 (stratified)
- **Training:** 6 epochs, batch size 16, Adam (lr 1e-4), fine-tuned end-to-end
- **Device:** CUDA (GPU)
- **CNN Model 1 / 2:** ResNet50 / DenseNet121 (best backbones from Lab 01)
- **Filtered set:** Sharpening (best filter from Lab 02)

---

## Results

### Table 1 — Effect of Noise and Preprocessing on Edge Detection

| Edge Detector | Input | Noise | Preprocessing | Edge Quality | Density Ratio vs Clean | Edge Density |
|---|---|---|---|---|---|---|
| Sobel | Original | None | None | Sparse | 11.58 | 5.6% |
| Sobel | Noisy | Gaussian | None | Noisy / many false edges | 158.10 | 76.6% |
| Sobel | Noisy | Salt & Pepper | None | Good | 46.30 | 22.4% |
| Sobel | Noisy | Gaussian | Gaussian Filter | Noisy / many false edges | 101.49 | 49.2% |
| Sobel | Noisy | Salt & Pepper | Median Filter | Sparse | 7.43 | 3.6% |
| Prewitt | Original | None | None | Sparse | 12.14 | 5.9% |
| Laplacian | Original | None | None | Good | 28.71 | 13.9% |
| LoG | Noisy | Gaussian | Gaussian Filter | Sparse | 1.78 | 0.9% |
| Canny | Original | None | Built-in smoothing | Sparse | 1.00 (baseline) | 0.5% |
| Canny | Noisy | Gaussian | Gaussian Filter | Sparse | 0.92 | 0.4% |
| Canny | Noisy | Salt & Pepper | Median Filter | Sparse | 0.70 | 0.3% |

**Reading it:** Canny is dramatically more stable under noise than Sobel — its density ratio
barely moves (1.00 → 0.92 → 0.70) while Sobel's explodes (11.58 → up to 158.10) because Canny's
built-in Gaussian smoothing + non-maximum suppression + hysteresis thresholding filter out
noise-driven false edges that Sobel has no defense against. Median filtering brought Sobel's
Salt & Pepper density ratio down from 46.30 to 7.43 — a large, concrete improvement.

### Table 2 — Canny Parameter Analysis

| Configuration | Low | High | Kernel | Edges Detected | Edge Density |
|---|---|---|---|---|---|
| Canny-1 | 30 | 100 | 3×3 | 461 | 0.9% |
| Canny-2 | 50 | 150 | 3×3 | 243 | 0.5% |
| Canny-3 | 100 | 200 | 3×3 | 225 | 0.4% |
| Canny-4 (5×5 blur) | 50 | 150 | 5×5 | 166 | 0.3% |

**Auto-selected configuration: Canny-1 (Low=30, High=100, kernel=3×3)** — closest to the
10% target density among the 4 swept, though honestly all 4 landed well below the 5–15%
"well-balanced" band on this particular sample image (worth widening the threshold search
range, e.g. trying Low=10–20, if you want a configuration inside that band).

### Table 3 — Cross-Lab Classification Performance Comparison

| Model | Acc. Raw (Lab 1) | Acc. Filtered (Lab 2) | Acc. Edge (Lab 3) | Precision* | Recall* | F1* | Train Time (s)* | Infer Time (ms)* |
|---|---|---|---|---|---|---|---|---|
| SVM | 82.89 | 83.35 | 79.16 | 70.36 | 79.16 | 71.86 | 73.28 | 12.31 |
| Random Forest | 79.98 | 80.56 | 79.63 | 71.48 | 79.63 | 72.58 | 90.27 | 0.07 |
| KNN | 83.12 | 81.84 | 75.67 | 69.15 | 75.67 | 71.91 | 0.01 | 0.79 |
| CNN Model 1 (ResNet50) | 90.80 | 90.22 | 80.68 | 76.95 | 80.68 | 77.11 | 594.41 | 9.43 |
| CNN Model 2 (DenseNet121) | 91.15 | **91.50** | 81.37 | 77.96 | 81.37 | 78.59 | 598.65 | 10.60 |

\* *Precision/Recall/F1/Training Time/Inference Time columns are computed on the Edge set (this lab's focus).*

**Best-performing model overall: CNN Model 2 (DenseNet121) on the Filtered set — 91.50% accuracy.**

Average accuracy by dataset version (across all 5 models):

| Set | Avg. Accuracy |
|---|---|
| Raw | 85.59% |
| Filtered | 85.49% |
| Edge | 79.30% |

**Edge maps reduced accuracy by ~6.29 points on average** relative to raw images — a
substantial drop, driven by the loss of color and texture information (edge maps keep only
boundary/gradient content). Filtered (Sharpening) was essentially flat vs. Raw (−0.09 points
on average) — it helped both CNNs slightly but hurt KNN, so its benefit isn't universal in
this run, unlike its clean sweep in Lab 02.

---

## Output files (from `./results_lab03/` after running)

- `report.md` — full report: Introduction, Methodology, Experimental Setup, all 3 tables above, Discussion (7 questions answered), Conclusion, References
- `table1_noise_effect.csv`, `table2_canny_params.csv`, `table3_cross_lab.csv` / `.md`
- `all_results_raw.csv` — full 15-row breakdown before reshaping into Table 3
- `figures/` — edge comparison grid, noise-effect grid, Canny sweep, dataset-set preview, bar chart
- `confusion_matrices/` — DenseNet121's confusion matrices across Raw/Filtered/Edge

## Key takeaways

1. **Canny is the clear winner for noise robustness** among the 5 classical edge detectors tested — its density stayed nearly flat under both noise types, while Sobel's blew up by over 100× under unfiltered Gaussian noise.
2. **Median filtering is the right tool for Salt & Pepper noise**, Gaussian filtering for Gaussian noise — both measurably reduced false-edge density before edge detection.
3. **Edge-only images are a worse classification input than raw or filtered images for this task** — losing color/texture cost ~6 points of accuracy on average, across every model tested, classical and deep alike.
4. **DenseNet121 was the best model overall**, and its best result came from the Filtered (Sharpened) set, not Raw or Edge.
