import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    """Base application configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'carepulse-healthcare-ml-secret-key-2026-production')
    
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'carepulse-jwt-super-secret-key-2026')
    JWT_TOKEN_LOCATION = ['cookies', 'headers']
    JWT_COOKIE_SECURE = os.environ.get('JWT_COOKIE_SECURE', 'False').lower() in ('true', '1')
    JWT_ACCESS_COOKIE_NAME = 'access_token_cookie'
    JWT_COOKIE_CSRF_PROTECT = False  # Keep false for frictionless clean form submissions in V1
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)
    
    # SQLite Database
    DATABASE_PATH = os.environ.get('DATABASE_PATH', os.path.join(basedir, 'database.db'))
    
    # ML and Recommendation Paths
    MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(basedir, 'models'))
    DATA_RECOMMENDATION_DIR = os.environ.get('DATA_RECOMMENDATION_DIR', os.path.join(basedir, 'data', 'recommendation'))
    DATA_RAW_DIR = os.path.join(basedir, 'data', 'raw')
    
    # Flask settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
    APP_NAME = "CarePulse Decision Support"
    
    # Healthcare Disclaimer
    DISCLAIMER = (
        "This system is designed for educational and research purposes. "
        "Machine-learning predictions are not a confirmed medical diagnosis, "
        "and recommendation information is not a medical prescription. "
        "Please consult a qualified healthcare professional for diagnosis, "
        "medication, treatment, or other medical decisions."
    )
