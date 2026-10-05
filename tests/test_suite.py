import os
import sys
import tempfile
import pytest
from werkzeug.security import generate_password_hash

# Ensure project root is in sys.path
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if basedir not in sys.path:
    sys.path.insert(0, basedir)

from app import create_app
from config import Config
from app.database.db import (
    init_db, get_db_connection, create_user, get_user_by_email,
    get_user_by_id, get_health_profile, upsert_health_profile,
    create_prediction, get_user_predictions, get_prediction_by_id,
    get_user_dashboard_stats
)
from app.services.prediction_service import prediction_service
from app.services.recommendation_service import recommendation_service

class TestConfig(Config):
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    # Use temporary database for isolation during testing
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    DATABASE_PATH = temp_db.name
    temp_db.close()

@pytest.fixture(scope='session')
def app():
    test_app = create_app(TestConfig)
    yield test_app
    # Clean up temp db
    if os.path.exists(TestConfig.DATABASE_PATH):
        try:
            os.remove(TestConfig.DATABASE_PATH)
        except Exception:
            pass

@pytest.fixture
def client(app):
    return app.test_client()

# --- 1. Database & User Isolation Tests ---

def test_database_initialization(app):
    db_path = app.config['DATABASE_PATH']
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row['name'] for row in cursor.fetchall()}
        assert 'users' in tables
        assert 'health_profiles' in tables
        assert 'predictions' in tables
        assert 'user_activity' in tables
        assert 'recommendation_feedback' in tables

def test_user_creation_and_isolation(app):
    db_path = app.config['DATABASE_PATH']
    u1_id = create_user(db_path, "Alice Patient", "alice@example.com", generate_password_hash("pass123"), 30, "Female")
    u2_id = create_user(db_path, "Bob Patient", "bob@example.com", generate_password_hash("pass456"), 45, "Male")
    
    assert u1_id != u2_id
    
    # Check health profiles auto-initialized
    p1 = get_health_profile(db_path, u1_id)
    p2 = get_health_profile(db_path, u2_id)
    assert p1['user_id'] == u1_id
    assert p2['user_id'] == u2_id
    assert p1['gender'] == "Female"
    assert p2['gender'] == "Male"
    
    # Create prediction for Alice
    pred1_id = create_prediction(
        db_path, u1_id, 30, "Female", "itching, skin_rash", "120/80", 95, 72, 180, None,
        "Random Forest", "Fungal infection", 0.95
    )
    
    # Alice can fetch her prediction
    alice_pred = get_prediction_by_id(db_path, pred1_id, u1_id)
    assert alice_pred is not None
    assert alice_pred['predicted_disease'] == "Fungal infection"
    
    # Bob MUST NOT be able to access Alice's prediction (Strict User Isolation)
    bob_attempt = get_prediction_by_id(db_path, pred1_id, u2_id)
    assert bob_attempt is None, "Security Violation: User 2 accessed User 1's prediction!"

# --- 2. Authentication Flow Tests ---

def test_auth_flows(client):
    # Test Signup with valid data
    res_signup = client.post('/signup', data={
        'name': 'Charlie Test',
        'email': 'charlie@example.com',
        'password': 'password123',
        'confirm_password': 'password123',
        'age': '28',
        'gender': 'Male'
    }, follow_redirects=True)
    assert res_signup.status_code == 200
    assert b"Welcome back, Charlie Test" in res_signup.data

    # Test Duplicate Email Signup Rejected
    res_dup = client.post('/signup', data={
        'name': 'Charlie Copy',
        'email': 'charlie@example.com',
        'password': 'password123',
        'confirm_password': 'password123',
        'age': '28',
        'gender': 'Male'
    })
    assert res_dup.status_code == 400
    assert b"already exists" in res_dup.data

    # Test Invalid Login
    res_bad_login = client.post('/login', data={
        'email': 'charlie@example.com',
        'password': 'wrongpassword'
    })
    assert res_bad_login.status_code == 401
    assert b"Invalid email or password" in res_bad_login.data

    # Test Valid Login
    res_login = client.post('/login', data={
        'email': 'charlie@example.com',
        'password': 'password123'
    }, follow_redirects=True)
    assert res_login.status_code == 200
    assert b"Welcome back" in res_login.data

    # Test Logout
    res_logout = client.get('/logout', follow_redirects=True)
    assert res_logout.status_code == 200
    assert b"logged out" in res_logout.data

def test_protected_routes_redirect_unauthenticated(client):
    # Unauthenticated requests to protected pages must redirect to /login
    for path in ['/dashboard', '/assessment', '/history', '/profile', '/result/1']:
        res = client.get(path)
        assert res.status_code == 302
        assert '/login' in res.headers['Location']

# --- 3. ML Model & Prediction Service Tests ---

def test_random_forest_prediction():
    # Random Forest should predict and provide probability
    res = prediction_service.predict('random_forest', ['itching', 'skin_rash', 'nodal_skin_eruptions'])
    assert res['model_name'] == "Random Forest"
    assert res['has_probability'] is True
    assert isinstance(res['probability_estimate'], float)
    assert res['probability_estimate'] > 0.5
    assert "Fungal infection" in res['predicted_disease']

def test_linear_svc_prediction():
    # Linear SVC should predict, and probability should be None
    res = prediction_service.predict('linear_svc', ['itching', 'skin_rash', 'nodal_skin_eruptions'])
    assert res['model_name'] == "Linear SVC"
    assert res['has_probability'] is False
    assert res['probability_estimate'] is None
    assert res['probability_percentage'] == "Not available for this model."
    assert "Fungal infection" in res['predicted_disease']

def test_invalid_model_choice():
    with pytest.raises(ValueError):
        prediction_service.predict('deep_neural_net', ['cough'])

def test_empty_symptoms_rejected():
    with pytest.raises(ValueError):
        prediction_service.predict('random_forest', [])

# --- 4. Knowledge-Based Recommendation Service Tests ---

def test_recommendation_retrieval():
    rec = recommendation_service.get_recommendations("Fungal infection")
    assert rec['disease'] == "Fungal infection"
    assert len(rec['description']) > 10
    assert len(rec['medicines']) > 0
    assert len(rec['precautions']) > 0
    assert len(rec['diets']) > 0
    assert len(rec['lifestyle']) > 0
    assert any("Antifungal" in m['name'] for m in rec['medicines'])

# --- 5. End-to-End Authenticated Assessment & Result Flow ---

def test_authenticated_assessment_and_result(client, app):
    db_path = app.config['DATABASE_PATH']
    # Login as Alice
    client.post('/login', data={'email': 'alice@example.com', 'password': 'pass123'})
    
    # Submit assessment with symptoms
    res_assessment = client.post('/assessment', data={
        'model_selection': 'random_forest',
        'age': '31',
        'gender': 'Female',
        'blood_pressure': '118/78',
        'glucose': '92',
        'heart_rate': '70',
        'cholesterol': '175',
        'symptoms': ['chills', 'cough', 'high_fever']
    }, follow_redirects=True)
    
    assert res_assessment.status_code == 200
    assert b"Prediction Result" in res_assessment.data
    assert b"Random Forest" in res_assessment.data
    assert b"Educational Recommendations" in res_assessment.data
    assert b"Print Report" in res_assessment.data

    # Verify history lists this assessment
    res_hist = client.get('/history')
    assert res_hist.status_code == 200
    assert b"Personal Prediction History" in res_hist.data
    assert b"Random Forest" in res_hist.data

    # Verify dashboard calculates real metrics
    res_dash = client.get('/dashboard')
    assert res_dash.status_code == 200
    assert b"Total Assessments" in res_dash.data
