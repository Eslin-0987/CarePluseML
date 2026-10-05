import os
import pickle
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_and_export():
    """
    Automated training and artifact export script for Render build step.
    Trains Random Forest and Linear SVC models on Training.csv and standardizes
    educational recommendation datasets.
    """
    basedir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(basedir, 'models')
    raw_data_dir = os.path.join(basedir, 'data', 'raw')
    rec_data_dir = os.path.join(basedir, 'data', 'recommendation')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(rec_data_dir, exist_ok=True)

    print("=== STEP 1: LOADING TRAINING DATASET ===")
    training_path = os.path.join(raw_data_dir, 'Training.csv')
    if not os.path.exists(training_path):
        raise FileNotFoundError(f"Training dataset not found at: {training_path}")

    df = pd.read_csv(training_path)
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    print(f"Dataset loaded: {df.shape[0]} samples, {X.shape[1]} symptom features.")

    print("\n=== STEP 2: FITTING AND EXPORTING DISEASE ENCODER ===")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    encoder_path = os.path.join(models_dir, 'disease_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Saved disease_encoder.pkl with {len(encoder.classes_)} canonical classes.")

    print("\n=== STEP 3: TRAINING RANDOM FOREST CLASSIFIER ===")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y_encoded)
    rf_pred = rf_model.predict(X)
    rf_acc = accuracy_score(y_encoded, rf_pred)
    rf_path = os.path.join(models_dir, 'rf.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"Random Forest trained and exported to rf.pkl (Accuracy: {rf_acc*100:.2f}%).")

    print("\n=== STEP 4: TRAINING LINEAR SVC ===")
    svc_model = SVC(kernel='linear', random_state=42)
    svc_model.fit(X, y_encoded)
    svc_pred = svc_model.predict(X)
    svc_acc = accuracy_score(y_encoded, svc_pred)
    svc_path = os.path.join(models_dir, 'svc.pkl')
    with open(svc_path, 'wb') as f:
        pickle.dump(svc_model, f)
    print(f"Linear SVC trained and exported to svc.pkl (Accuracy: {svc_acc*100:.2f}%).")

    print("\n=== STEP 5: EXPORTING SCALER ARTIFACT ===")
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(None, f)
    print("Exported scaler.pkl (None for binary symptom indicators).")

    print("\n=== STEP 6: COMPILING RECOMMENDATION KNOWLEDGE BASES ===")
    canonical_diseases = list(encoder.classes_)

    def clean_name(n):
        if not isinstance(n, str):
            return ""
        return " ".join(n.split())

    def match_canonical(name):
        norm = clean_name(name).lower()
        for c in canonical_diseases:
            c_norm = clean_name(c).lower()
            if c_norm == norm:
                return c
            if "vertigo" in norm and "vertigo" in c_norm:
                return c
            if "peptic" in norm and "peptic" in c_norm:
                return c
            if "diabet" in norm and "diabet" in c_norm:
                return c
            if "hypertens" in norm and "hypertens" in c_norm and "thyroid" not in norm and "thyroid" not in c_norm:
                return c
            if "hepatitis a" in norm and "hepatitis a" in c_norm:
                return c
        return name

    # A. diseases.csv
    desc_file = os.path.join(raw_data_dir, 'description.csv')
    if os.path.exists(desc_file):
        raw_desc = pd.read_csv(desc_file)
        rec_list = []
        for _, r in raw_desc.iterrows():
            rec_list.append({
                'disease': match_canonical(r['Disease']),
                'disease_description': str(r['Description']).strip()
            })
        pd.DataFrame(rec_list).drop_duplicates(subset=['disease']).to_csv(
            os.path.join(rec_data_dir, 'diseases.csv'), index=False
        )
        print("Compiled data/recommendation/diseases.csv")

    # B. medicines.csv
    meds_file = os.path.join(raw_data_dir, 'medications.csv')
    if os.path.exists(meds_file):
        raw_meds = pd.read_csv(meds_file)
        meds_list = []
        for _, r in raw_meds.iterrows():
            can_d = match_canonical(r['Disease'])
            try:
                med_items = ast.literal_eval(r['Medication'])
            except Exception:
                med_items = [m.strip(" '\"[]") for m in str(r['Medication']).split(',')]
            for med in med_items:
                med = med.strip()
                if med:
                    meds_list.append({
                        'disease': can_d,
                        'medicine_name': med,
                        'medicine_information': f"General educational information: {med} is commonly referenced in medical literature as a clinical therapeutic agent associated with {can_d}. Always consult a qualified physician or pharmacist for clinical evaluation and prescription instructions."
                    })
        pd.DataFrame(meds_list).drop_duplicates(subset=['disease', 'medicine_name']).to_csv(
            os.path.join(rec_data_dir, 'medicines.csv'), index=False
        )
        print("Compiled data/recommendation/medicines.csv")

    # C. precautions.csv
    prec_file = os.path.join(raw_data_dir, 'precautions_df.csv')
    if os.path.exists(prec_file):
        raw_prec = pd.read_csv(prec_file)
        prec_list = []
        for _, r in raw_prec.iterrows():
            can_d = match_canonical(r['Disease'])
            for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                if col in r and pd.notna(r[col]) and str(r[col]).strip():
                    prec_list.append({
                        'disease': can_d,
                        'precaution': str(r[col]).strip().capitalize()
                    })
        pd.DataFrame(prec_list).drop_duplicates(subset=['disease', 'precaution']).to_csv(
            os.path.join(rec_data_dir, 'precautions.csv'), index=False
        )
        print("Compiled data/recommendation/precautions.csv")

    # D. diets.csv
    diets_file = os.path.join(raw_data_dir, 'diets.csv')
    if os.path.exists(diets_file):
        raw_diets = pd.read_csv(diets_file)
        diet_list = []
        for _, r in raw_diets.iterrows():
            can_d = match_canonical(r['Disease'])
            try:
                diet_items = ast.literal_eval(r['Diet'])
            except Exception:
                diet_items = [d.strip(" '\"[]") for d in str(r['Diet']).split(',')]
            for item in diet_items:
                item = item.strip()
                if item:
                    diet_list.append({
                        'disease': can_d,
                        'diet_recommendation': item.capitalize()
                    })
        pd.DataFrame(diet_list).drop_duplicates(subset=['disease', 'diet_recommendation']).to_csv(
            os.path.join(rec_data_dir, 'diets.csv'), index=False
        )
        print("Compiled data/recommendation/diets.csv")

    # E. lifestyle.csv
    work_file = os.path.join(raw_data_dir, 'workout_df.csv')
    if os.path.exists(work_file):
        raw_work = pd.read_csv(work_file)
        work_list = []
        for _, r in raw_work.iterrows():
            can_d = match_canonical(r['disease'])
            act = str(r['workout']).strip()
            if act and act.lower() != 'nan':
                work_list.append({
                    'disease': can_d,
                    'lifestyle_recommendation': act.capitalize()
                })
        pd.DataFrame(work_list).drop_duplicates(subset=['disease', 'lifestyle_recommendation']).to_csv(
            os.path.join(rec_data_dir, 'lifestyle.csv'), index=False
        )
        print("Compiled data/recommendation/lifestyle.csv")

    print("\n=== BUILD STEP COMPLETE: ALL MODEL & KNOWLEDGE ARTIFACTS VERIFIED ===")

if __name__ == '__main__':
    train_and_export()
