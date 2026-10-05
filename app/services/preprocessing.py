import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Exact 132 features in exact order expected by the trained Random Forest and Linear SVC models
FEATURE_ORDER = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
    'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
    'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
    'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
    'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches',
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
    'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# Set of valid feature names for fast O(1) lookup
VALID_FEATURE_SET = set(FEATURE_ORDER)

def clean_feature_label(feature_name: str) -> str:
    """Format technical feature names into user-friendly clinical labels."""
    # Special customized labels for clear healthcare presentation
    custom_labels = {
        'spotting_ urination': 'Spotting During Urination',
        'cold_hands_and_feets': 'Cold Hands and Feet',
        'foul_smell_of urine': 'Foul Smell of Urine',
        'continuous_feel_of_urine': 'Continuous Urge to Urinate',
        'toxic_look_(typhos)': 'Toxic Look (Typhoid Appearance)',
        'dischromic _patches': 'Discolored / Pigmented Skin Patches',
        'fluid_overload.1': 'Systemic Fluid Overload',
        'scurring': 'Skin Scarring / Scabbing',
        'silver_like_dusting': 'Silvery Flaking / Dusting',
        'receiving_unsterile_injections': 'History of Injections with Unsterile Equipment',
        'receiving_blood_transfusion': 'History of Blood Transfusion',
        'extra_marital_contacts': 'History of High-Risk / Multiple Sexual Contacts',
        'history_of_alcohol_consumption': 'History of Regular Alcohol Consumption',
        'family_history': 'Family History of Chronic Disease',
        'altered_sensorium': 'Altered Consciousness / Confusion',
        'spinning_movements': 'Sensation of Spinning (Vertigo)',
        'pus_filled_pimples': 'Pus-Filled Pimples / Pustules',
        'red_sore_around_nose': 'Red Sores Around Nose or Mouth',
        'yellow_crust_ooze': 'Yellow Crust / Oozing Lesions',
        'drying_and_tingling_lips': 'Drying and Tingling Lips',
        'weakness_of_one_body_side': 'Weakness on One Side of the Body'
    }
    if feature_name in custom_labels:
        return custom_labels[feature_name]
    
    # Generic cleanup: replace underscores with spaces and capitalize words
    clean = feature_name.replace('_', ' ').strip()
    return ' '.join(word.capitalize() for word in clean.split())

# Categorized taxonomy for grouping symptoms cleanly in the UI
CATEGORY_MAPPING = {
    'General & Constitutional': [
        'fatigue', 'high_fever', 'mild_fever', 'chills', 'shivering', 'sweating',
        'malaise', 'headache', 'dizziness', 'weight_loss', 'weight_gain',
        'lethargy', 'restlessness', 'dehydration', 'anxiety', 'mood_swings', 'depression', 'irritability'
    ],
    'Skin, Hair & Nails': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'yellowish_skin', 'bruising',
        'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'red_spots_over_body',
        'dischromic _patches', 'brittle_nails', 'small_dents_in_nails', 'inflammatory_nails'
    ],
    'Respiratory & ENT': [
        'continuous_sneezing', 'cough', 'breathlessness', 'phlegm', 'throat_irritation',
        'sinus_pressure', 'runny_nose', 'congestion', 'patches_in_throat', 'loss_of_smell',
        'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum'
    ],
    'Digestive & Abdominal': [
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'indigestion',
        'nausea', 'loss_of_appetite', 'constipation', 'abdominal_pain', 'diarrhoea',
        'belly_pain', 'passage_of_gases', 'stomach_bleeding', 'distention_of_abdomen',
        'swelling_of_stomach', 'acute_liver_failure'
    ],
    'Musculoskeletal & Neurological': [
        'joint_pain', 'muscle_wasting', 'back_pain', 'weakness_in_limbs', 'neck_pain',
        'cramps', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
        'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
        'unsteadiness', 'weakness_of_one_body_side', 'slurred_speech', 'altered_sensorium',
        'lack_of_concentration', 'muscle_pain', 'painful_walking', 'coma'
    ],
    'Eyes & Vision': [
        'sunken_eyes', 'yellowing_of_eyes', 'pain_behind_the_eyes',
        'blurred_and_distorted_vision', 'redness_of_eyes', 'visual_disturbances',
        'watering_from_eyes', 'puffy_face_and_eyes'
    ],
    'Urinary & Excretory': [
        'burning_micturition', 'spotting_ urination', 'dark_urine', 'yellow_urine',
        'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'polyuria',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'
    ],
    'Cardiovascular & Circulation': [
        'chest_pain', 'fast_heart_rate', 'palpitations', 'swollen_legs',
        'swollen_blood_vessels', 'swollen_extremeties', 'prominent_veins_on_calf',
        'fluid_overload', 'fluid_overload.1', 'cold_hands_and_feets'
    ],
    'Endocrine & Medical Background': [
        'irregular_sugar_level', 'increased_appetite', 'excessive_hunger', 'obesity',
        'enlarged_thyroid', 'abnormal_menstruation', 'swelled_lymph_nodes',
        'toxic_look_(typhos)', 'extra_marital_contacts', 'family_history',
        'receiving_blood_transfusion', 'receiving_unsterile_injections',
        'history_of_alcohol_consumption', 'drying_and_tingling_lips', 'internal_itching'
    ]
}

def get_categorized_symptoms():
    """
    Return a structured catalog of all 132 features grouped by clinical category,
    suitable for rendering the interactive health assessment form.
    """
    catalog = {}
    assigned_features = set()
    
    for category, feature_list in CATEGORY_MAPPING.items():
        catalog[category] = []
        for feat in feature_list:
            if feat in VALID_FEATURE_SET:
                catalog[category].append({
                    'id': feat,
                    'label': clean_feature_label(feat)
                })
                assigned_features.add(feat)
    
    # Catch any features not explicitly assigned and group them into 'Other Symptoms'
    remaining = [f for f in FEATURE_ORDER if f not in assigned_features]
    if remaining:
        catalog['Other Symptoms'] = [{'id': f, 'label': clean_feature_label(f)} for f in remaining]
        
    return catalog

def get_all_symptoms_flat():
    """Return flat list of all 132 symptoms with id, label, and category for client-side search."""
    flat_list = []
    categorized = get_categorized_symptoms()
    for cat, items in categorized.items():
        for item in items:
            flat_list.append({
                'id': item['id'],
                'label': item['label'],
                'category': cat
            })
    return flat_list

def prepare_features(selected_symptoms_list, scaler=None):
    """
    Transform user-selected symptoms into the exact 132-dimension feature array
    with correct column ordering and binary indicators.
    
    Parameters:
        selected_symptoms_list (list): List of symptom string keys selected by user.
        scaler: Optional pre-fitted scaler. If None, binary features are used directly.
        
    Returns:
        pd.DataFrame: A 1-row DataFrame with 132 columns in FEATURE_ORDER.
        list: List of valid recognized symptoms included.
    """
    if not isinstance(selected_symptoms_list, (list, set, tuple)):
        raise ValueError("selected_symptoms_list must be an iterable of strings.")
    
    selected_set = {str(s).strip() for s in selected_symptoms_list if str(s).strip()}
    
    # Find matching valid symptoms
    recognized_symptoms = [s for s in selected_set if s in VALID_FEATURE_SET]
    unrecognized = selected_set - VALID_FEATURE_SET
    if unrecognized:
        logger.warning(f"Ignored unrecognized symptoms: {unrecognized}")
        
    # Construct binary feature vector
    row_values = [1 if feature in selected_set else 0 for feature in FEATURE_ORDER]
    features_df = pd.DataFrame([row_values], columns=FEATURE_ORDER)
    
    # Apply saved scaler if present
    if scaler is not None:
        scaled_values = scaler.transform(features_df)
        features_df = pd.DataFrame(scaled_values, columns=FEATURE_ORDER)
        
    return features_df, recognized_symptoms
