import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class RecommendationService:
    """
    Knowledge-Based / Rule-Based Recommendation System.
    Loads educational clinical knowledge datasets from CSV files and maps
    predicted conditions to relevant educational records.
    """
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.diseases_df = None
        self.medicines_df = None
        self.precautions_df = None
        self.diets_df = None
        self.lifestyle_df = None
        self.is_loaded = False
        
        if data_dir:
            self.load_data(data_dir)

    def load_data(self, data_dir):
        """Load and index recommendation CSV files into memory."""
        self.data_dir = data_dir
        logger.info(f"Loading recommendation datasets from: {data_dir}")
        
        diseases_path = os.path.join(data_dir, 'diseases.csv')
        medicines_path = os.path.join(data_dir, 'medicines.csv')
        precautions_path = os.path.join(data_dir, 'precautions.csv')
        diets_path = os.path.join(data_dir, 'diets.csv')
        lifestyle_path = os.path.join(data_dir, 'lifestyle.csv')

        for path in [diseases_path, medicines_path, precautions_path, diets_path, lifestyle_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required recommendation dataset missing: {path}")

        self.diseases_df = pd.read_csv(diseases_path)
        self.medicines_df = pd.read_csv(medicines_path)
        self.precautions_df = pd.read_csv(precautions_path)
        self.diets_df = pd.read_csv(diets_path)
        self.lifestyle_df = pd.read_csv(lifestyle_path)
        
        self.is_loaded = True
        logger.info("Recommendation service initialized with 5 knowledge databases.")
        return True

    def _normalize_name(self, name: str) -> str:
        """Helper to normalize disease strings for matching."""
        if not name:
            return ""
        s = " ".join(str(name).strip().lower().split())
        s = s.replace("diseae", "disease")
        s = s.replace("  ", " ")
        return s

    def get_recommendations(self, disease_name: str) -> dict:
        """
        Retrieve educational recommendations for the specified predicted condition.
        
        Returns a dictionary with:
            - disease: str
            - description: str
            - medicines: list of dicts [{'name': ..., 'information': ...}]
            - precautions: list of str
            - diets: list of str
            - lifestyle: list of str
        """
        if not self.is_loaded:
            raise RuntimeError("RecommendationService data not loaded.")

        target_norm = self._normalize_name(disease_name)
        
        # 1. Disease Description
        desc_matches = self.diseases_df[
            self.diseases_df['disease'].apply(self._normalize_name) == target_norm
        ]
        if not desc_matches.empty:
            description = desc_matches.iloc[0]['disease_description']
            matched_canonical = desc_matches.iloc[0]['disease']
        else:
            # Substring fallback
            desc_sub = self.diseases_df[
                self.diseases_df['disease'].apply(lambda d: self._normalize_name(d) in target_norm or target_norm in self._normalize_name(d))
            ]
            if not desc_sub.empty:
                description = desc_sub.iloc[0]['disease_description']
                matched_canonical = desc_sub.iloc[0]['disease']
            else:
                description = (
                    f"{disease_name} is a medical condition documented in clinical literature. "
                    "Please consult a licensed healthcare professional for comprehensive diagnostic workup."
                )
                matched_canonical = disease_name

        matched_norm = self._normalize_name(matched_canonical)

        # 2. General Medicine Information (Educational only)
        meds_matches = self.medicines_df[
            self.medicines_df['disease'].apply(self._normalize_name) == matched_norm
        ]
        medicines = []
        for _, row in meds_matches.iterrows():
            medicines.append({
                'name': str(row['medicine_name']).strip(),
                'information': str(row['medicine_information']).strip()
            })

        # 3. Precautions
        prec_matches = self.precautions_df[
            self.precautions_df['disease'].apply(self._normalize_name) == matched_norm
        ]
        precautions = []
        for _, row in prec_matches.iterrows():
            prec = str(row['precaution']).strip()
            if prec and prec not in precautions:
                precautions.append(prec)

        # 4. Diet Recommendations
        diet_matches = self.diets_df[
            self.diets_df['disease'].apply(self._normalize_name) == matched_norm
        ]
        diets = []
        for _, row in diet_matches.iterrows():
            diet_item = str(row['diet_recommendation']).strip()
            if diet_item and diet_item not in diets:
                diets.append(diet_item)

        # 5. Lifestyle & Supportive Measures
        life_matches = self.lifestyle_df[
            self.lifestyle_df['disease'].apply(self._normalize_name) == matched_norm
        ]
        lifestyle = []
        for _, row in life_matches.iterrows():
            act = str(row['lifestyle_recommendation']).strip()
            if act and act not in lifestyle:
                lifestyle.append(act)

        return {
            'disease': disease_name,
            'canonical_disease': matched_canonical,
            'description': description,
            'medicines': medicines,
            'precautions': precautions,
            'diets': diets,
            'lifestyle': lifestyle
        }

# Global singleton instance
recommendation_service = RecommendationService()
