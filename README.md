# ANN Customer Churn Prediction

A Jupyter Notebook implementation of a binary classification Artificial Neural Network (ANN) to predict bank customer churn. Built with TensorFlow/Keras on the classic Churn Modelling dataset, with full preprocessing, model training, EarlyStopping, TensorBoard logging, and artifact serialization.

---

## Problem Statement

Predict whether a bank customer will churn (leave the bank) — a binary classification task where `Exited = 1` means churned and `Exited = 0` means retained.

---

## Dataset

**File:** `Churn_Modelling.csv`  
**Rows:** 10,000 customers  
**Source:** Standard bank churn benchmark dataset

| Column | Description |
|---|---|
| `RowNumber` | Row index (dropped) |
| `CustomerId` | Unique customer ID (dropped) |
| `Surname` | Customer surname (dropped) |
| `CreditScore` | Customer credit score |
| `Geography` | Country: France / Germany / Spain |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Credit card holder (0/1) |
| `IsActiveMember` | Active member (0/1) |
| `EstimatedSalary` | Estimated annual salary |
| **`Exited`** | **Target — churned (1) or retained (0)** |

---

## Pipeline Overview

```
Churn_Modelling.csv
        │
        ▼
  Drop irrelevant cols (RowNumber, CustomerId, Surname)
        │
        ├── LabelEncoder  ──► Gender (Male/Female → 0/1)
        │
        └── OneHotEncoder ──► Geography → Geography_France, Geography_Germany, Geography_Spain
                │
                ▼
        Concatenate back into feature matrix (12 features total)
                │
                ▼
        Train/Test Split (80/20, random_state=42)
                │
                ▼
        StandardScaler (fit on train, transform both)
                │
                ▼
        ANN Model (TensorFlow/Keras)
        ┌──────────────────┐
        │ Dense(64, ReLU)  │  ← Hidden Layer 1
        │ Dense(32, ReLU)  │  ← Hidden Layer 2
        │ Dense(1, Sigmoid)│  ← Output Layer
        └──────────────────┘
                │
                ▼
     Adam (lr=0.01) + BinaryCrossentropy
                │
                ▼
     EarlyStopping + TensorBoard callbacks
                │
                ▼
     Saved: model.h5, *.pkl files
```

---

## Model Architecture

| Layer | Type | Units | Activation | Params |
|---|---|---|---|---|
| Input | Dense | 64 | ReLU | 832 |
| Hidden | Dense | 32 | ReLU | 2,080 |
| Output | Dense | 1 | Sigmoid | 33 |
| **Total** | | | | **2,945** |

---

## Tech Stack

| Component | Library |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Preprocessing | scikit-learn (`LabelEncoder`, `OneHotEncoder`, `StandardScaler`) |
| Serialization | `pickle` |
| Visualization | TensorBoard |
| Python Version | 3.13 |

---

## Prerequisites

- Python 3.10+
- Anaconda environment (recommended)
- `Churn_Modelling.csv` placed at the path referenced in the notebook

---

## Installation

```bash
pip install tensorflow scikit-learn pandas numpy
```

---

## Usage

1. Place `Churn_Modelling.csv` in your working directory (or update the path in cell 2).
2. Open `ann.ipynb` in Jupyter and run all cells in order.
3. The notebook will:
   - Preprocess and encode the data
   - Split into train/test sets (80/20)
   - Scale features using `StandardScaler`
   - Build and compile the ANN
   - Train with EarlyStopping and TensorBoard callbacks
   - Save the trained model and preprocessing artifacts

### Launch TensorBoard

```bash
tensorboard --logdir log/fit
```

Then open `http://localhost:6006` in your browser to monitor training curves.

---

## Output Files

| File | Description |
|---|---|
| `model.h5` | Trained Keras ANN model (HDF5 format) |
| `scaler.pkl` | Fitted `StandardScaler` for inference |
| `label_encoder_gender.pkl` | Fitted `LabelEncoder` for Gender |
| `onehot_encoder_Geo.pkl` | Fitted `OneHotEncoder` for Geography |
| `log/fit/` | TensorBoard training logs |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Loss | Binary Crossentropy |
| Metrics | Accuracy |
| Epochs (max) | 100 |
| Batch Size | 32 (default) |
| Validation Split | 20% (X_test / y_test) |
| Early Stopping | `patience=5`, monitors `val_loss` |
| Best Weights Restored | Yes |

---

## Training Results

Training stopped early at **epoch 16** (early stopping triggered). Sample results:

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|---|---|---|---|
| 1 | 83.3% | 84.9% | 0.3657 |
| 7 | 86.1% | 86.3% | 0.3397 |
| 11 | 86.8% | 86.3% | 0.3388 ✓ best |
| 16 | 87.1% | 86.4% | 0.3574 |

---

## Project Structure

```
.
├── ann.ipynb                     # Main notebook
├── Churn_Modelling.csv           # Input dataset (provide separately)
├── model.h5                      # Saved trained model
├── scaler.pkl                    # Saved StandardScaler
├── label_encoder_gender.pkl      # Saved LabelEncoder (Gender)
├── onehot_encoder_Geo.pkl        # Saved OneHotEncoder (Geography)
└── log/
    └── fit/                      # TensorBoard logs
```

---

## Notes

- **Model format:** The model is saved as `model.h5` (legacy HDF5). Keras recommends using the native `.keras` format going forward: `model.save('model.keras')`.
- **GPU on Windows:** TensorFlow ≥ 2.11 does not support native Windows GPU. Training runs on CPU. Use WSL2 or the TensorFlow-DirectML plugin for GPU acceleration.
- **TensorBoard kernel crash:** A kernel crash was observed when launching `%tensorboard` inline in VS Code. Run TensorBoard from the terminal instead (`tensorboard --logdir log/fit`) to avoid this.
- **Inference:** To make predictions on new data, load all four saved artifacts (`model.h5`, `scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_Geo.pkl`) and apply the same preprocessing pipeline before calling `model.predict()`.

---

## Inference Example

```python
import pickle
import numpy as np
import tensorflow as tf

# Load artifacts
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f: le_gender = pickle.load(f)
with open('onehot_encoder_Geo.pkl', 'rb') as f: ohe_geo = pickle.load(f)

# Sample customer: [CreditScore, Gender, Age, Tenure, Balance,
#                   NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography]
gender_enc = le_gender.transform(['Male'])[0]
geo_enc = ohe_geo.transform([['France']]).toarray()[0]
features = np.array([[600, gender_enc, 40, 3, 60000, 2, 1, 1, 50000, *geo_enc]])
features_scaled = scaler.transform(features)

probability = model.predict(features_scaled)[0][0]
print(f"Churn probability: {probability:.2%}")
```
