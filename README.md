# Credit Card Fraud Detection Machine Learning Web Application

A professional, responsive machine learning web application that classifies credit card transactions in real-time as **Legitimate** or **Fraudulent**. This project translates an end-to-end Machine Learning research pipeline into an enterprise-grade web application using **Flask**, **Scikit-Learn**, **Pandas**, and **Bootstrap 5**.

---

## 1. Project Title
**Credit Card Fraud Detection Machine Learning Web Application**  
*Real-Time Anomaly & Financial Risk Inference System*

---

## 2. Project Overview
Credit card fraud presents a significant challenge for modern banking and payment processing systems, costing the financial industry billions of dollars annually. Because fraudulent transactions occur at an extremely low frequency relative to legitimate ones (less than 0.2% in realistic settings), standard classification methods struggle with extreme class imbalance.

This project implements an end-to-end solution:
1. Addresses extreme class imbalance using a **stratified random undersampling strategy** based strictly on the experimental notebook design.
2. Fits a **Logistic Regression** classifier over 30 principal component and numerical transaction features.
3. Serves the trained model via a production-grade **Flask web application** featuring a dashboard UI with real-time inference, dual-layer validation, sample presets, and prediction probability confidence metrics.

---

## 3. Features
- **Strict Pipeline Replication:** Replicates the data sampling, train/test split, feature ordering, and model architecture from the reference notebook.
- **30-Feature Numerical Input Form:** Cleanly groups inputs into 4 logical sections:
  - *Transaction Information:* `Time`, `Amount`
  - *Principal Components:* `V1` – `V10`
  - *Principal Components:* `V11` – `V20`
  - *Principal Components:* `V21` – `V28`
- **1-Click Verified Sample Presets:** Instant loading of real benchmark legitimate and fraudulent transaction vectors for demonstration without manual typing.
- **Dynamic Dual-Path Inference:**
  - Asynchronous AJAX submission via `/predict` JSON API for instant real-time UI updates without page reloads.
  - Server-side fallback via traditional form POST rendering.
- **Distinct Visual Classification Feedback:**
  - **Legitimate Transaction:** Emerald green card with shield check and verified safety status.
  - **Fraudulent Transaction Detected:** High-visibility crimson alert card with risk warning badge.
- **Real-World Risk Analytics & Banking Action Protocol:**
  - Computes a dynamic **Financial Risk Score (0 to 100)**.
  - Automatically recommends banking actions:
    - `APPROVE` (Score $< 30$): Instant transaction authorization.
    - `STEP_UP_OTP` (Score $30 - 70$): Flags for 3D-Secure Two-Factor Authentication.
    - `BLOCK` (Score $\ge 70$): Immediate decline and cardholder anomaly alert.
- **Explainable AI (XAI) & Feature Contribution Waterfall:**
  - Decomposes the Logistic Regression logit score:
    $$z = \beta_0 + \sum_{i=1}^{30} \beta_i X_i$$
  - Identifies **Top Risk Drivers** pushing the transaction toward Fraud with contribution values ($\beta_i X_i > 0$), feature values, and financial anomaly descriptions (e.g. $V14$ primary behavioral anomaly, $V4$ velocity vector).
  - Identifies **Top Safety Factors** pulling the transaction toward Legitimate status ($\beta_i X_i < 0$).
  - 1-Click **"Copy Audit Summary"** to export compliance records.

---

## 4. Machine Learning Algorithm
- **Algorithm:** **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- **Optimization Solver:** L-BFGS (`lbfgs`) with `max_iter=1000`
- **Decision Function:** Standard Sigmoid / Logit link function:
  $$P(Y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{30} \beta_i X_i)}}$$
- **Classification Criterion:**
  - $P(Y = 1 | X) \ge 0.5 \implies \text{Class } 1 \text{ (Fraudulent Transaction)}$
  - $P(Y = 1 | X) < 0.5 \implies \text{Class } 0 \text{ (Legitimate Transaction)}$
- **Underlying Rationale:** Logistic Regression offers high interpretability, rapid sub-millisecond inference times suitable for high-throughput payment gateways, and calibrated posterior probabilities via `predict_proba()`.

---

## 5. Dataset Description
The model is trained on the standardized **Credit Card Fraud Detection Dataset**:
- **Total Features:** 30 numerical input features + 1 target variable (`Class`).
- **Feature Set:**
  - `Time`: Elapsed time in seconds between this transaction and the first transaction in the dataset.
  - `V1` to `V28`: 28 numerical features resulting from a **Principal Component Analysis (PCA)** transformation applied to protect sensitive user details and banking privacy.
  - `Amount`: The monetary value of the transaction.
- **Target Variable:**
  - `Class = 0`: Legitimate Transaction.
  - `Class = 1`: Fraudulent Transaction.
- **Class Balancing Method:**
  - Legitimate transactions are downsampled (`n = 492`) to match the fraud instances, yielding a balanced dataset of 756 to 984 transactions, preventing majority class bias.

---

## 6. Technology Stack

### Backend
- **Python 3.10+ / 3.14**
- **Flask (v3.0+):** Lightweight WSGI web application framework and RESTful API router.
- **Scikit-Learn (v1.3+):** Model training, train/test splitting, and accuracy evaluation.
- **Pandas (v2.0+):** DataFrame manipulation, feature ordering, and dataset balancing.
- **NumPy (v1.24+):** Numerical computations and multidimensional array conversions.
- **Joblib (v1.3+):** High-performance model serialization and persistence (`fraud_model.pkl`).

### Frontend
- **HTML5:** Semantic, accessible markup.
- **CSS3:** Custom fintech dark theme with responsive CSS variables, smooth transitions, and distinct status glowing borders.
- **JavaScript (Vanilla ES6+):** Async `fetch` API, DOM manipulation, and input validation.
- **Bootstrap 5.3 & Bootstrap Icons:** Responsive mobile-first grid, navigation bar, and glyphs.

---

## 7. Project Structure
```text
credit-card-fraud/
│
├── app.py                # Main Flask web application and prediction API
├── train_model.py        # Model training script reproducing the notebook
├── fraud_model.pkl       # Trained Logistic Regression model binary
├── model_metrics.json    # Serialized model accuracy and training metadata
├── requirements.txt      # Python package dependencies
├── README.md             # Comprehensive documentation and interview guide
│
├── templates/
│   └── index.html        # Responsive ML dashboard template
│
└── static/
    ├── style.css         # Modern fintech dark theme styling
    └── script.js         # Interactive form handling and async inference
```

---

## 8. Installation Instructions

### Step 1: Open a Terminal / PowerShell
Navigate to the project root directory:
```bash
cd "C:\Users\sistl\.gemini\antigravity\scratch\credit-card-fraud"
```

### Step 2: (Recommended) Create and Activate a Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

---

## 9. How to Train the Model
The model has already been trained and saved as `fraud_model.pkl`. To retrain the model at any time:

```bash
python train_model.py
```

### Advanced Training Flags:
- **Specify a custom dataset path:**
  ```bash
  python train_model.py --data-path "path/to/creditcard.csv"
  ```
- **Train on the full Kaggle dataset balanced (492 legit vs 492 fraud):**
  ```bash
  python train_model.py --mode full
  ```
- **Default notebook mode (first 140,703 rows):**
  ```bash
  python train_model.py --mode notebook
  ```

During execution, `train_model.py` outputs the training and test accuracy scores, confusion matrix, and classification report, and updates `fraud_model.pkl` and `model_metrics.json`.

---

## 10. How to Run the Flask Application
Start the Flask development server by running:

```bash
python app.py
```

Expected output:
```text
============================================================
  CREDIT CARD FRAUD DETECTION WEB APPLICATION
  Running at: http://127.0.0.1:5000
============================================================
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

Open your web browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 11. How to Deploy to Render (Cloud Hosting)

The application includes production configuration for **[Render](https://render.com)** (`gunicorn`, `render.yaml`, `Procfile`, and dynamic port binding).

### Step 1: Push Code to GitHub
1. Create a new repository on GitHub (e.g., `credit-card-fraud-detection`).
2. Link your local project and push:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/credit-card-fraud-detection.git
   git push -u origin main
   ```

### Step 2: Deploy on Render
- **Option A — 1-Click Blueprint (Recommended):**
  1. Log into your [Render Dashboard](https://dashboard.render.com).
  2. Click **New +** > **Blueprint**.
  3. Select your `credit-card-fraud-detection` repository.
  4. Render will automatically read `render.yaml`, configure Python, and deploy.
  5. Click **Apply**.

- **Option B — Standard Web Service:**
  1. Click **New +** > **Web Service**.
  2. Select your repository.
  3. Configure settings:
     - **Name:** `credit-card-fraud-detection`
     - **Runtime:** `Python 3`
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `gunicorn app:app`
     - **Plan:** `Free`
  4. Click **Create Web Service**.

Render will deploy the application to a public HTTPS URL (e.g., `https://credit-card-fraud-detection-xxxx.onrender.com`).

---

## 12. How to Use the Prediction Form

1. **Option A — Quick Benchmark Testing (Recommended):**
   - Click the **Legit Sample** button in the form header to pre-fill all 30 features with verified legitimate transaction data.
   - Click **Predict Transaction**. The system displays a green **Legitimate Transaction** card with model confidence and probability meters.
   - Click the **Fraud Sample** button in the form header to pre-fill all 30 features with a verified fraudulent transaction pattern.
   - Click **Predict Transaction**. The system displays a red **Fraudulent Transaction Detected** card with fraud probability.
2. **Option B — Manual Feature Entry:**
   - Enter values into the **Transaction Information** section (`Time` and `Amount`).
   - Fill in the PCA principal components across the sections:
     - `V1` to `V10`
     - `V11` to `V20`
     - `V21` to `V28`
   - Click **Predict Transaction**.
3. **Resetting:**
   - Click the **Reset** button to clear all input fields, dismiss previous warnings, and hide the prediction card.

---

## 12. Model Accuracy & Evaluation Metrics

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Test Accuracy** | **~95.39%** | Evaluated on the 20% stratified holdout split |
| **Training Accuracy** | **~96.36%** | Evaluated on the 80% training split |
| **Precision (Legit)** | **0.98** | High precision minimizing false alarms |
| **Recall (Fraud)** | **0.96** | 96% detection rate on fraudulent transactions |
| **F1-Score (Fraud)** | **0.94** | Harmonic mean of fraud precision & recall |

### Test Split Confusion Matrix
```text
                  Predicted Legit (0)    Predicted Fraud (1)
Actual Legit (0)          94                      5
Actual Fraud (1)           2                     51
```

---

## 13. Limitations & Real-World Considerations

1. **Undersampling Information Loss:**
   - Random undersampling balances the class distribution effectively for linear models, but discards non-fraud training instances. In a high-volume banking environment, hybrid techniques (SMOTE, SMOTE-ENN, or Cost-Sensitive XGBoost) may be preferred.
2. **Feature Anonymization:**
   - The features `V1` through `V28` are anonymized PCA projections. While PCA preserves variance and eliminates multicollinearity, raw domain features (such as IP address velocity, geolocation mismatch, device fingerprinting, and merchant category codes) are abstracted.
3. **Concept Drift:**
   - Fraud patterns evolve over time as adversaries adapt. Production systems require continuous retraining pipelines and concept drift monitoring.
4. **Threshold Calibration:**
   - In production banking, the decision threshold (default $0.5$) can be tuned based on business risk appetites to optimize for recall over false positives.
