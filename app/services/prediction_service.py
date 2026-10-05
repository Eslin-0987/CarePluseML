import os
import pickle
import logging
import numpy as np
import pandas as pd
from app.services.preprocessing import prepare_features, clean_feature_label

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Production machine-learning prediction service for disease decision-support.
    Safely loads trained Random Forest and Linear SVC models and their artifacts.
    """
    def __init__(self, models_dir=None):
        self.models_dir = models_dir
        self.rf_model = None
        self.svc_model = None
        self.disease_encoder = None
        self.scaler = None
        self.is_loaded = False
        
        if models_dir:
            self.load_models(models_dir)

    def load_models(self, models_dir):
        """Load serialized model files and verify their interfaces."""
        self.models_dir = models_dir
        logger.info(f"Loading machine learning models from: {models_dir}")
        
        rf_path = os.path.join(models_dir, 'rf.pkl')
        svc_path = os.path.join(models_dir, 'svc.pkl')
        encoder_path = os.path.join(models_dir, 'disease_encoder.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')

        # 1. Load Random Forest model
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"Required model file not found: {rf_path}")
        with open(rf_path, 'rb') as f:
            self.rf_model = pickle.load(f)
        logger.info(f"Loaded Random Forest model. Features: {getattr(self.rf_model, 'n_features_in_', 'unknown')}")

        # 2. Load Linear SVC model
        if not os.path.exists(svc_path):
            raise FileNotFoundError(f"Required model file not found: {svc_path}")
        with open(svc_path, 'rb') as f:
            self.svc_model = pickle.load(f)
        logger.info(f"Loaded Linear SVC model (kernel={getattr(self.svc_model, 'kernel', 'linear')}). Features: {getattr(self.svc_model, 'n_features_in_', 'unknown')}")

        # 3. Load Disease Label Encoder
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Required encoder file not found: {encoder_path}")
        with open(encoder_path, 'rb') as f:
            self.disease_encoder = pickle.load(f)
        logger.info(f"Loaded disease encoder with {len(self.disease_encoder.classes_)} classes.")

        # 4. Load Scaler if present
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                if self.scaler is not None:
                    logger.info("Loaded pre-trained feature scaler.")
                else:
                    logger.info("Scaler loaded as None (binary indicators do not require scaling).")
            except Exception as e:
                logger.warning(f"Could not load scaler (proceeding without scaler): {e}")
                self.scaler = None
        else:
            self.scaler = None

        self.is_loaded = True
        return True

    def predict(self, model_key: str, symptoms_list: list) -> dict:
        """
        Execute prediction on user symptom inputs using the selected trained model.
        
        Parameters:
            model_key (str): 'random_forest' or 'linear_svc'
            symptoms_list (list): list of string symptom IDs
            
        Returns:
            dict containing:
                - model_key: str
                - model_name: str
                - model_description: str
                - predicted_disease: str (decoded canonical name)
                - probability_estimate: float or None
                - probability_percentage: str or None
                - has_probability: bool
                - recognized_symptoms: list of dicts with id and label
                - features_active_count: int
        """
        if not self.is_loaded:
            raise RuntimeError("PredictionService models are not loaded. Call load_models() first.")
            
        if not symptoms_list or len(symptoms_list) == 0:
            raise ValueError("At least one symptom must be selected to generate a prediction.")

        # Prepare feature vector (1, 132)
        features_df, recognized_symptoms = prepare_features(symptoms_list, scaler=self.scaler)
        
        if len(recognized_symptoms) == 0:
            raise ValueError("None of the submitted symptoms match known model features. Please review selection.")

        # Select model
        normalized_key = model_key.lower().strip()
        if normalized_key in ('random_forest', 'rf'):
            model = self.rf_model
            model_display_name = "Random Forest"
            model_description = "Ensemble decision-tree model with 100 estimators. Evaluates voting across randomized decision trees."
            
            # Predict class
            raw_prediction = model.predict(features_df)[0]
            
            # Probability estimate
            if hasattr(model, 'predict_proba'):
                proba_array = model.predict_proba(features_df)[0]
                # Find probability for the predicted class index
                # model.classes_ contains the class indices [0, 1, ..., 40]
                class_index = np.where(model.classes_ == raw_prediction)[0]
                if len(class_index) > 0:
                    prob_val = float(proba_array[class_index[0]])
                else:
                    prob_val = float(np.max(proba_array))
                probability_estimate = round(prob_val, 4)
                probability_percentage = f"{round(prob_val * 100, 1)}%"
                has_probability = True
            else:
                probability_estimate = None
                probability_percentage = "Not available for this model."
                has_probability = False
                
        elif normalized_key in ('linear_svc', 'svc', 'svm'):
            model = self.svc_model
            model_display_name = "Linear SVC"
            model_description = "Support Vector Classifier with a linear kernel: SVC(kernel='linear'). Identifies optimal hyperplanes for multiclass separation."
            
            # Predict class
            raw_prediction = model.predict(features_df)[0]
            
            # Linear SVC has probability=False; predict_proba is not available.
            # Never invent or simulate a fake percentage!
            probability_estimate = None
            probability_percentage = "Not available for this model."
            has_probability = False
            
        else:
            raise ValueError(f"Invalid model selection '{model_key}'. Allowed choices: 'random_forest' or 'linear_svc'.")

        # Decode disease label using disease_encoder
        try:
            if hasattr(self.disease_encoder, 'inverse_transform'):
                predicted_disease_raw = self.disease_encoder.inverse_transform([raw_prediction])[0]
            else:
                predicted_disease_raw = str(raw_prediction)
        except Exception as e:
            logger.error(f"Error decoding disease label: {e}")
            predicted_disease_raw = str(raw_prediction)

        # Clean display name for the predicted condition
        # 'Peptic ulcer diseae' -> 'Peptic ulcer disease'
        # '(vertigo) Paroymsal  Positional Vertigo' -> '(vertigo) Paroxysmal Positional Vertigo'
        clean_condition = predicted_disease_raw.strip()
        if clean_condition.lower() == "peptic ulcer diseae":
            clean_condition = "Peptic Ulcer Disease"
        elif "vertigo" in clean_condition.lower():
            clean_condition = "(Vertigo) Paroxysmal Positional Vertigo"
        elif clean_condition.lower() == "diabetes ":
            clean_condition = "Diabetes"
        elif clean_condition.lower() == "hypertension ":
            clean_condition = "Hypertension"
        elif clean_condition.lower() == "hepatitis a":
            clean_condition = "Hepatitis A"

        # Format recognized symptoms for user-facing review
        symptoms_details = [
            {'id': s, 'label': clean_feature_label(s)} for s in recognized_symptoms
        ]

        return {
            'model_key': normalized_key,
            'model_name': model_display_name,
            'model_description': model_description,
            'predicted_disease': clean_condition,
            'raw_predicted_disease': predicted_disease_raw,
            'probability_estimate': probability_estimate,
            'probability_percentage': probability_percentage,
            'has_probability': has_probability,
            'recognized_symptoms': symptoms_details,
            'features_active_count': len(recognized_symptoms)
        }

# Global singleton instance for application use
prediction_service = PredictionService()
