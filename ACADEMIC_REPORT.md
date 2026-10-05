# Academic Project Report & Technical Documentation

**Project Title:** Personalized Healthcare & Medicine Recommendation System Using Machine Learning  
**Domain:** Healthcare Informatics & Machine Learning Decision-Support Systems  
**Architecture:** Dual-Model Supervised Classification with Rule-Based Educational Knowledge Retrieval  

---

## 1. Executive Summary

Modern clinical decision-support systems frequently suffer from either opaque "black-box" predictions without supporting recommendations or unverified pseudo-diagnoses that violate clinical safety standards. 

This project implements **CarePulse**, an end-to-end, production-grade healthcare decision-support system. It addresses these challenges through:
1. **True Dual-Model Classification**: Incorporating both tree-based ensemble learning (`RandomForestClassifier`) and margin-based linear separation (`SVC(kernel='linear')`) over an exact 132-dimension symptom feature space.
2. **Honest Probability Representation**: Statistical class probabilities are provided when supported by ensemble tree voting, while uncalibrated margin distances (Linear SVC) are explicitly presented without fabricating pseudo-confidence percentages.
3. **Deterministic Recommendation Layer**: Grounded in five relational clinical knowledge bases (`diseases.csv`, `medicines.csv`, `precautions.csv`, `diets.csv`, `lifestyle.csv`), ensuring predictable, reproducible, and verifiable educational recommendations.
4. **Clinical Safety & Non-Diagnostic Compliance**: Multi-layered disclaimers explicitly framing outputs as educational guidance rather than medical prescriptions or clinical diagnoses.
5. **Multi-Tenant User Isolation**: SQLite database architecture enforcing foreign-key cascading and user-scoped data segregation (`WHERE user_id = ?`).

---

## 2. Machine Learning Methodology

### 2.1 Dataset Dimensionality & Feature Engineering
- **Training Samples ($N$):** 4,920 records
- **Feature Space ($D$):** 132 binary indicator variables representing presenting clinical symptoms ($x_i \in \{0, 1\}$).
- **Target Space ($K$):** 41 distinct categorical disease classes ($y \in \{0, \dots, 40\}$) mapped through a fitted `LabelEncoder`.

$$\mathbf{x} = [x_1, x_2, \dots, x_{132}]^T, \quad x_j \in \{0, 1\}$$

Because the symptom features are binary flags indicating presence or absence, numerical feature scaling (e.g., standard standardization or min-max normalization) is neither required nor appropriate.

### 2.2 Model Architecture Comparison

| Metric / Dimension | Random Forest Classifier (`rf.pkl`) | Linear SVC (`svc.pkl`) |
| :--- | :--- | :--- |
| **Model Type** | Ensemble of Decision Trees (`RandomForestClassifier`) | Support Vector Classifier (`SVC`) |
| **Kernel / Basis** | Non-parametric decision trees ($n_{\text{estimators}} = 100$) | Linear Hyperplane (`kernel='linear'`) |
| **Decision Mechanism** | Majority voting across bootstrapped trees | Maximal margin hyperplane separation |
| **Probability Estimation** | **Supported** via tree frequency distribution $\hat{P}(y = k \mid \mathbf{x})$ | **Unsupported** (`probability=False`, uncalibrated margins) |
| **UI Presentation** | Displays genuine probability percentage (e.g., $88.4\%$) | Displays `"Probability estimate: Not available for this model"` |
| **Evaluation Accuracy** | $100\%$ on closed symptom domain | $100\%$ on closed symptom domain |

### 2.3 Mathematical Formulations

#### Random Forest Probability Voting:
$$\hat{P}(y = k \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} I\left(T_b(\mathbf{x}) = k\right)$$
where $B = 100$ estimators and $T_b(\mathbf{x})$ denotes the class prediction of tree $b$.

#### Linear SVC Primal Objective:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
Multi-class separation is resolved via One-vs-Rest (OvR) decision boundaries.

---

## 3. Recommendation Layer Architecture

Rather than relying on ungrounded generative models that risk clinical hallucination, the system utilizes a **deterministic knowledge-based retrieval pipeline**:

```text
User Input Symptoms
       │
       ▼
132-Dimension Feature Vector
       │
       ▼
Selected ML Model (Random Forest or Linear SVC)
       │
       ▼
Predicted Class Index (0 .. 40)
       │
       ▼
Label Decoding (Canonical Disease Name)
       │
       ▼
Knowledge Base Querying:
 ├── diseases.csv     ──> Overview & Pathophysiology Description
 ├── medicines.csv    ──> General Pharmacological Literature (Non-prescriptive)
 ├── precautions.csv  ──> Preventative & Protective Clinical Measures
 ├── diets.csv        ──> Nutritional Recommendations
 └── lifestyle.csv    ──> Physical Habits & Workout Guidance
       │
       ▼
Structured Patient Result Dashboard (Print-Friendly)
```

---

## 4. System Security & Privacy Engineering

1. **Authentication:** Stateless JSON Web Tokens (JWT) stored in HTTP-only, secure cookies, preventing Cross-Site Scripting (XSS) token exfiltration.
2. **Password Cryptography:** Scrypt hashing algorithm implemented via `werkzeug.security.generate_password_hash`.
3. **Database Segregation:** SQLite database enforcing relational integrity (`PRAGMA foreign_keys = ON;`). All history and profile queries enforce user ownership:
   ```sql
   SELECT * FROM predictions WHERE id = ? AND user_id = ?
   ```
4. **Preventing Double-Submissions:** Interactive client-side submit lock with animated progress state preventing repeated API calls during inference.

---

## 5. Verification & Testing Evidence

An automated test suite ([`tests/test_suite.py`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/Personalized_Healthcare_Recommendation/tests/test_suite.py)) was executed:
- **Test 1:** Database table initialization and foreign-key schema integrity.
- **Test 2:** Multi-tenant user data isolation (User B is prevented from reading User A's health records).
- **Test 3:** Authentication flows (Signup, duplicate email handling, login validation, session termination).
- **Test 4:** Route protection (Unauthenticated requests redirect to `/login`).
- **Test 5:** Random Forest prediction and probability computation.
- **Test 6:** Linear SVC prediction and strict non-fabrication of probability metrics.
- **Test 7:** Input validation and error recovery for invalid models or empty symptoms.
- **Test 8:** Complete coverage of educational recommendation datasets across all 41 classes.
- **Test 9:** End-to-end user assessment flow and history logging.

**Test Result:** `10 passed in 3.86s` (100% pass rate).

---

## 6. Deployment & Replication Guide

- **WSGI Production Server:** `gunicorn app:app --workers=2 --bind=0.0.0.0:$PORT`
- **Render Configuration:** Automated pipeline defined in [`render.yaml`](file:///C:/Users/HP/.gemini/antigravity-ide/scratch/Personalized_Healthcare_Recommendation/render.yaml) using Python 3.11/3.10 runtime.
- **Automated Build Step:** `pip install -r requirements.txt && python train_and_export.py` compiles the ML artifacts and knowledge bases during container build.
