# CarePulse: Personalized Healthcare & Medicine Recommendation System Using Machine Learning

A production-ready, secure, transparent healthcare machine-learning decision-support web application.

> **Healthcare Safety Notice:**
> This system is designed strictly for educational and research purposes. Machine-learning predictions are not a confirmed medical diagnosis, and recommendation information is not a medical prescription. Please consult a qualified healthcare professional for diagnosis, medication, treatment, or other medical decisions.

---

## 1. Project Overview & Architecture

CarePulse integrates actual trained machine learning models (**Random Forest Classifier** and **Linear Support Vector Classifier**) with a knowledge-based recommendation engine and SQLite database persistence.

### Key Capabilities
- **Dual ML Classification**: Choose between Random Forest (`rf.pkl`, with probability estimates) and Linear SVC (`svc.pkl`, `SVC(kernel='linear')`, with strict non-simulated probability handling).
- **Exact 132 Clinical Features**: The assessment form is grounded directly in the 132 binary symptom indicators extracted from model training metadata.
- **Rule-Based Educational Recommendations**: Curated knowledge bases for Disease Overview, Educational Medicine Information, Precautions, Dietary Guidance, and Lifestyle/Workout Supportive Measures.
- **JWT & Password Security**: Secure password hashing with Scrypt, HTTP-only JWT authentication cookies, and server-side validation.
- **Strict Data Isolation**: Multi-tenant database schema ensuring users can only query and view their own personal health assessments and prediction histories.
- **Render Ready**: Includes `Procfile`, `render.yaml`, environment variable handling, and Gunicorn WSGI configuration.

---

## 2. Directory Structure

```text
Personalized_Healthcare_Recommendation/
│
├── app.py                      # Flask WSGI entrypoint
├── config.py                   # Environment & application configuration
├── requirements.txt            # Python dependencies
├── database.db                 # SQLite database with foreign keys & WAL
├── Procfile                    # Render/Heroku WSGI deployment command
├── render.yaml                 # Render Blueprint specification
├── .env                        # Environment secrets
├── .gitignore                  # Git ignore rules
├── README.md                   # System documentation
│
├── models/                     # Serialized ML artifacts
│   ├── rf.pkl                  # Trained Random Forest model (100 estimators)
│   ├── svc.pkl                 # Trained Linear SVC model (kernel='linear')
│   ├── scaler.pkl              # Scaler placeholder (binary features need no scaling)
│   └── disease_encoder.pkl     # LabelEncoder for 41 canonical disease classes
│
├── data/
│   ├── raw/                    # Original training & metadata CSV files
│   ├── processed/              # Cleaned datasets
│   └── recommendation/         # Standardized educational knowledge bases
│       ├── diseases.csv        # Disease descriptions
│       ├── medicines.csv       # General medication literature entries
│       ├── precautions.csv     # Recommended protective precautions
│       ├── diets.csv           # Supportive nutritional guidance
│       └── lifestyle.csv       # Lifestyle and workout recommendations
│
├── notebooks/
│   └── disease_prediction.ipynb # Training, evaluation, and serialization notebook
│
├── app/
│   ├── __init__.py             # Flask application factory
│   ├── routes/
│   │   ├── auth.py             # Signup, Login, Logout, JWT auth decorators
│   │   ├── dashboard.py        # Real calculated metrics, health summary
│   │   ├── prediction.py       # Health assessment form, predictions, history
│   │   └── recommendation.py   # Profile editing, methodology, research feedback
│   │
│   ├── services/
│   │   ├── preprocessing.py    # 132-dimension feature vector alignment & categorization
│   │   ├── prediction_service.py # Model loading, inference, probability evaluation
│   │   └── recommendation_service.py # In-memory knowledge base indexing & retrieval
│   │
│   └── database/
│       └── db.py               # Parameterized SQLite queries & user isolation
│
├── templates/
│   ├── base.html               # Shared accessible application shell
│   ├── login.html              # Authentication login
│   ├── signup.html             # Profile registration
│   ├── dashboard.html          # User dashboard with real database stats
│   ├── assessment.html         # Interactive symptom checklist with real-time search
│   ├── result.html             # Prediction focal point & educational cards (print-ready)
│   ├── history.html            # User-isolated prediction history (table & cards)
│   ├── profile.html            # Identity & clinical vitals management
│   └── about.html              # Technical methodology & research roadmap
│
├── static/
│   ├── css/                    # Modular component stylesheets
│   │   ├── base.css
│   │   ├── login.css
│   │   ├── signup.css
│   │   ├── dashboard.css
│   │   ├── assessment.css
│   │   ├── result.css
│   │   ├── history.css
│   │   ├── profile.css
│   │   └── about.css
│   ├── js/
│   │   ├── main.js             # Navigation, alerts, password toggles
│   │   └── assessment.js       # Symptom search, category filtering, chip tray
│   └── images/
│       └── logo.png            # CarePulse healthcare brand logo
│
└── tests/
    └── test_suite.py           # Automated pytest validation suite (10/10 passed)
```

---

## 3. Local Installation & Execution

### Prerequisites
- Python 3.10+
- pip

### Step-by-Step Setup
1. Clone or navigate to the project directory:
   ```bash
   cd Personalized_Healthcare_Recommendation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the automated test suite:
   ```bash
   pytest tests/test_suite.py
   ```

4. Launch the local development server:
   ```bash
   python app.py
   ```
   Access the web application in your browser at: `http://localhost:5000`

---

## 4. Machine Learning & Preprocessing Specifications

- **Feature Space**: 132 binary indicators representing presenting symptoms.
- **Feature Preservation**: `app/services/preprocessing.py` strictly guarantees feature column ordering matching `rf.feature_names_in_`.
- **Target Classes**: 41 distinct medical conditions decoded through `disease_encoder.pkl`.
- **Probability Handling**:
  - **Random Forest**: Supported via ensemble tree voting (`predict_proba`). Displayed strictly as a statistical probability estimate.
  - **Linear SVC**: `SVC(kernel='linear')` uses uncalibrated margin distance. Probability estimates are explicitly declared as **"Not available for this model"** without fabricating artificial percentages.

---

## 5. Security & Isolation Controls

- **Password Hashing**: Uses `scrypt` hashing via Werkzeug Security.
- **JWT Protection**: Tokens stored in secure HTTP-only cookies with configurable expiration.
- **Database Query Parameterization**: All SQL queries use SQLite `?` placeholders preventing SQL injection.
- **Record Ownership Enforcement**: All historical and profile queries include `WHERE user_id = ?`, forbidding cross-user data leakage.
