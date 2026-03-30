# Customer Churn Prediction with ANN

Predicting whether a bank customer will leave using a simple feedforward neural network. Built this to get hands-on with the full deep learning workflow — preprocessing, model building, callbacks, and saving artifacts for reuse.

---

## What it does

Takes customer data (age, balance, geography, credit score, etc.) and predicts churn probability as a binary classification problem. The model outputs a value between 0 and 1 — closer to 1 means higher churn risk.

Dataset is the standard Churn Modelling CSV (10,000 rows, 14 columns).

---

## Preprocessing

A couple of things worth noting here:

- **Gender** gets label encoded (Male/Female → 0/1)
- **Geography** gets one-hot encoded into three columns (France, Germany, Spain) since it's a multi-class nominal feature — didn't want to imply any ordinal relationship
- Dropped `RowNumber`, `CustomerId`, `Surname` — purely identifiers, no signal
- Standard scaled everything before feeding into the network

All encoders and the scaler are saved as pickle files so inference stays consistent with training.

---

## Model

Pretty straightforward architecture — two hidden layers with ReLU, sigmoid output:

```
Input (12 features)
    → Dense(64, ReLU)
    → Dense(32, ReLU)
    → Dense(1, Sigmoid)
```

Compiled with Adam (lr=0.01) and binary crossentropy. Added EarlyStopping on `val_loss` with patience=5 and `restore_best_weights=True` so it doesn't overfit on a longer run.

Training stopped at epoch 16, best val accuracy around **86.4%**.

---

## TensorBoard

Logs are written to `log/fit/`. To view:

```bash
tensorboard --logdir log/fit
```

> **Note:** Launching `%tensorboard` inline in VS Code caused a kernel crash. Running it from terminal works fine.

---

## Saved Files

After running the notebook you'll have:

```
model.h5
scaler.pkl
label_encoder_gender.pkl
onehot_encoder_Geo.pkl
```

Load all four for inference — skipping any of them will cause a preprocessing mismatch.

---

## Quick Inference

```python
import pickle
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f: le = pickle.load(f)
with open('onehot_encoder_Geo.pkl', 'rb') as f: ohe = pickle.load(f)

gender = le.transform(['Male'])[0]
geo = ohe.transform([['France']]).toarray()[0]
sample = np.array([[600, gender, 40, 3, 60000, 2, 1, 1, 50000, *geo]])

prob = model.predict(scaler.transform(sample))[0][0]
print(f"Churn probability: {prob:.2%}")
```

---

## Stack

Python 3.13 · TensorFlow/Keras · scikit-learn · Pandas · NumPy
