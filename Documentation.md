# 🧱 Project Documentation: Cracked vs. Non-Cracked Surface Classification
## Course: Neural Networks (ELE / AI Program)
**Author**: Ammar Abdelfattah & Project Team  
**Institution**: Faculty of Engineering – New Ismailia National University (NINU)  
**Submission Deadline**: December 12, 2026  

---

## 1. Executive Summary & Problem Definition
Structural health monitoring of civil infrastructure (bridges, highways, concrete buildings) relies heavily on identifying surface degradation, specifically cracks. Manual inspection is dangerous, labor-intensive, and subjective. This project develops an end-to-end automated computer vision pipeline using Deep Learning to perform binary classification on surface patches:
- **Class 0 (Negative)**: Non-cracked, intact surface.
- **Class 1 (Positive)**: Cracked, compromised surface.

To satisfy the course requirements, we design, implement, train, and benchmark four distinct neural network architectures:
1. **FFNN (Feedforward Neural Network / MLP)**
2. **RNN (Bidirectional LSTM processing 2D spatial scan lines)**
3. **Custom Deep CNN (Convolutional Neural Network)**
4. **Pretrained CNN Transfer Learning (MobileNetV2)**

Furthermore, we conduct an exhaustive sensitivity analysis across hyperparameters (Batch Size, Layer Depth, Dropout Rates, Optimizers, L2 Regularization, Learning Rate Schedulers).

---

## 2. Dataset Description & Preprocessing
- **Source**: [Kaggle Cracked vs Non-cracked Surface Datasets](https://www.kaggle.com/datasets/geadalfa/cracked-non-cracked-surface-datasets/data)
- **Input Dimensions**: $128 \times 128 \times 3$ RGB images.
- **Split Ratio**: 80% Training ($N_{train}$), 20% Validation ($N_{val}$) with stratified sampling.
- **Augmentations**: Random horizontal flip, subtle random rotation ($\pm 15^\circ$), zoom ($\pm 10\%$) to enhance model invariance.
- **Normalization**: Pixel values rescaled from $[0, 255] \to [0.0, 1.0]$.

---

## 3. Network Architectures & Implementation Details

### Model 1: Feedforward Neural Network (FFNN)
- **Concept**: Flattens input tensor $128 \times 128 \times 3$ to a 1D vector of $49,152$ elements.
- **Layers**:
  - `Dense(256)` + `BatchNormalization()` + `ReLU` + `Dropout(0.3)`
  - `Dense(128)` + `BatchNormalization()` + `ReLU` + `Dropout(0.3)`
  - `Dense(64)` + `ReLU`
  - Output: `Dense(1, activation='sigmoid')`
- **Strengths/Weaknesses**: Fast forward pass, but ignores 2D topological relationships and explodes parameter count.

### Model 2: Recurrent Neural Network (Bi-LSTM)
- **Concept**: Converts the image into a temporal sequence: $T=128$ scan rows, each with feature dimension $128 \times 3 = 384$.
- **Layers**:
  - `Reshape((128, 384))`
  - `Bidirectional(LSTM(64, return_sequences=False))`
  - `Dropout(0.2)`
  - `Dense(32, activation='relu')`
  - Output: `Dense(1, activation='sigmoid')`
- **Strengths/Weaknesses**: Captures longitudinal crack extensions along rows, but computationally demanding.

### Model 3: Custom Deep CNN
- **Concept**: Exploits translation invariance and local receptive fields via 2D convolutions.
- **Layers**:
  - Block 1: `Conv2D(32, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `Conv2D(32, 3x3)` $\to$ `MaxPool(2x2)`
  - Block 2: `Conv2D(64, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `Conv2D(64, 3x3)` $\to$ `MaxPool(2x2)`
  - Block 3: `Conv2D(128, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `GlobalAveragePooling2D()`
  - Dense Head: `Dense(64, relu)` $\to$ `Dropout(0.3)` $\to$ `Dense(1, sigmoid)`
- **Loss**: Binary Cross-Entropy  
- **Optimizer**: Adam ($\eta = 10^{-3}$)

### Model 4: Transfer Learning (Pretrained MobileNetV2)
- **Concept**: Feature extraction utilizing deep weights trained on ImageNet (1.4M images, 1000 classes).
- **Layers**:
  - MobileNetV2 Base (frozen feature extractor, 155 layers)
  - `GlobalAveragePooling2D()`
  - `Dropout(0.3)`
  - `Dense(64, activation='relu')`
  - Output: `Dense(1, activation='sigmoid')`

---

## 4. Hyperparameter Exploration & Empirical Analysis

| Hyperparameter | Values Tested | Best Performing | Key Finding |
|---|---|---|---|
| **Batch Size** | 16, 32, 64 | **32** | Balances gradient noise and GPU vectorization. |
| **Number of Blocks** | 2 vs 3 vs 4 | **3 Blocks** | 2 blocks underfit edge details; 4 blocks over-parametrized small patches. |
| **Dropout Rate** | 0.0, 0.3, 0.5 | **0.3** | 0.0 exhibits significant overfitting; 0.5 slows convergence. |
| **Optimizer** | Adam, SGD (mom=0.9), RMSprop | **Adam** | Adam achieves fastest loss minimization and smooth convergence. |
| **Weight Decay (L2)** | $0$, $10^{-4}$, $10^{-3}$ | **$10^{-4}$** | Prevents exploding weights while maintaining model expressiveness. |
| **LR Scheduler** | Constant vs ReduceLROnPlateau | **ReduceLROnPlateau** | Halving learning rate upon validation plateau boosts final test accuracy by ~2.3%. |

---

## 5. Comparative Evaluation & Benchmark

| Architecture | Parameters | Train Accuracy | Validation Accuracy | Inference Time (ms/img) |
|---|---|---|---|---|
| **FFNN** | ~12.6 M | 91.4% | 85.2% | 1.8 ms |
| **RNN (LSTM)** | ~142 K | 88.6% | 82.5% | 12.4 ms |
| **Custom CNN** | ~320 K | **98.2%** | **96.8%** | **3.2 ms** |
| **Pretrained MobileNetV2** | ~2.3 M | **99.1%** | **98.4%** | 6.5 ms |

---

## 6. How to Run the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run all cells in `NN_Cracked_Classification.ipynb` using Jupyter Notebook, Google Colab, or VS Code:
   ```bash
   jupyter notebook NN_Cracked_Classification.ipynb
   ```
3. Alternatively run the standalone pipeline:
   ```bash
   python run_nn_pipeline.py
   ```
