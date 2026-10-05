import sqlite3
import os
from datetime import datetime

def get_db_connection(db_path):
    """Create and configure a robust, concurrent SQLite connection with WAL mode."""
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
    except Exception:
        pass
    return conn

def init_db(db_path):
    """Initialize database tables according to specification."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            age INTEGER,
            gender TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # 2. health_profiles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            age INTEGER,
            gender TEXT,
            blood_pressure TEXT,
            glucose REAL,
            heart_rate INTEGER,
            cholesterol REAL,
            medical_history TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        """)
        
        # 3. predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age INTEGER,
            gender TEXT,
            symptoms TEXT,
            blood_pressure TEXT,
            glucose REAL,
            heart_rate INTEGER,
            cholesterol REAL,
            medical_history TEXT,
            model_used TEXT NOT NULL,
            predicted_disease TEXT NOT NULL,
            prediction_probability REAL,
            prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        """)
        
        # 4. user_activity table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity_type TEXT NOT NULL,
            activity_details TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        """)
        
        # 5. recommendation_feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            recommendation_type TEXT,
            item_name TEXT,
            feedback TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        """)
        
        conn.commit()

# --- User Repository Functions ---

def create_user(db_path, name, email, password_hash, age=None, gender=None):
    """Create a new user and initialize an empty health profile."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """INSERT INTO users (name, email, password_hash, role, age, gender, created_at, updated_at)
               VALUES (?, ?, ?, 'user', ?, ?, ?, ?)""",
            (name.strip(), email.strip().lower(), password_hash, age, gender, now, now)
        )
        user_id = cursor.lastrowid
        
        # Initialize corresponding health profile record
        cursor.execute(
            """INSERT INTO health_profiles (user_id, age, gender, updated_at)
               VALUES (?, ?, ?, ?)""",
            (user_id, age, gender, now)
        )
        conn.commit()
        return user_id

def get_user_by_email(db_path, email):
    """Fetch user by email."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_user_by_id(db_path, user_id):
    """Fetch user by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, role, age, gender, created_at, updated_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def update_user_account(db_path, user_id, name, age=None, gender=None):
    """Update user identity information."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """UPDATE users SET name = ?, age = ?, gender = ?, updated_at = ? WHERE id = ?""",
            (name.strip(), age, gender, now, user_id)
        )
        conn.commit()

def update_user_password(db_path, user_id, password_hash):
    """Update user password hash."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?""",
            (password_hash, now, user_id)
        )
        conn.commit()

# --- Health Profile Functions ---

def get_health_profile(db_path, user_id):
    """Fetch user health profile."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM health_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def upsert_health_profile(db_path, user_id, age=None, gender=None, blood_pressure=None, 
                          glucose=None, heart_rate=None, cholesterol=None, medical_history=None):
    """Insert or update user health profile."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """INSERT INTO health_profiles (user_id, age, gender, blood_pressure, glucose, heart_rate, cholesterol, medical_history, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   age=excluded.age,
                   gender=excluded.gender,
                   blood_pressure=excluded.blood_pressure,
                   glucose=excluded.glucose,
                   heart_rate=excluded.heart_rate,
                   cholesterol=excluded.cholesterol,
                   medical_history=excluded.medical_history,
                   updated_at=excluded.updated_at""",
            (user_id, age, gender, blood_pressure, glucose, heart_rate, cholesterol, medical_history, now)
        )
        conn.commit()

# --- Predictions Functions (Always User-Isolated) ---

def create_prediction(db_path, user_id, age, gender, symptoms, blood_pressure, 
                      glucose, heart_rate, cholesterol, medical_history, 
                      model_used, predicted_disease, prediction_probability):
    """Record a prediction in the database."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """INSERT INTO predictions 
               (user_id, age, gender, symptoms, blood_pressure, glucose, heart_rate, cholesterol, medical_history, model_used, predicted_disease, prediction_probability, prediction_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, age, gender, symptoms, blood_pressure, glucose, heart_rate, cholesterol, medical_history, model_used, predicted_disease, prediction_probability, now)
        )
        pred_id = cursor.lastrowid
        conn.commit()
        return pred_id

def get_user_predictions(db_path, user_id, limit=None, search=None, model_filter=None, sort_order="DESC"):
    """Fetch user's own predictions with optional search, filtering, and limit."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM predictions WHERE user_id = ?"
        params = [user_id]
        
        if search:
            query += " AND (predicted_disease LIKE ? OR symptoms LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
            
        if model_filter and model_filter != 'all':
            query += " AND model_used = ?"
            params.append(model_filter)
            
        order_direction = "ASC" if sort_order.upper() == "ASC" else "DESC"
        query += f" ORDER BY prediction_date {order_direction}"
        
        if limit and isinstance(limit, int):
            query += f" LIMIT {limit}"
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_prediction_by_id(db_path, prediction_id, user_id):
    """Fetch a single prediction, verifying that it belongs to the authenticated user."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
            (prediction_id, user_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

def get_user_dashboard_stats(db_path, user_id):
    """
    Calculate real metrics for the user dashboard:
    - total_predictions
    - latest_prediction
    - last_assessment_date
    - most_used_model
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Total predictions
        cursor.execute("SELECT COUNT(*) AS total FROM predictions WHERE user_id = ?", (user_id,))
        total_predictions = cursor.fetchone()['total']
        
        # 2. Latest prediction
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC LIMIT 1",
            (user_id,)
        )
        latest_row = cursor.fetchone()
        latest_prediction = dict(latest_row) if latest_row else None
        
        # 3. Most used model
        cursor.execute(
            """SELECT model_used, COUNT(*) AS count 
               FROM predictions 
               WHERE user_id = ? 
               GROUP BY model_used 
               ORDER BY count DESC 
               LIMIT 1""",
            (user_id,)
        )
        model_row = cursor.fetchone()
        most_used_model = model_row['model_used'] if model_row else "None yet"
        
        return {
            "total_predictions": total_predictions,
            "latest_prediction": latest_prediction,
            "last_assessment_date": latest_prediction['prediction_date'] if latest_prediction else None,
            "most_used_model": most_used_model
        }

# --- Activity Logging ---

def log_activity(db_path, user_id, activity_type, activity_details=None):
    """Log an activity event for an authenticated user without blocking requests."""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                """INSERT INTO user_activity (user_id, activity_type, activity_details, created_at)
                   VALUES (?, ?, ?, ?)""",
                (user_id, activity_type, activity_details, now)
            )
            conn.commit()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Non-critical activity log failed: {e}")

def get_user_activity(db_path, user_id, limit=8):
    """Retrieve recent user activities."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM user_activity 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ?""",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# --- Recommendation Feedback ---

def add_feedback(db_path, user_id, recommendation_type, item_name, feedback):
    """Record user feedback for future research."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """INSERT INTO recommendation_feedback (user_id, recommendation_type, item_name, feedback, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, recommendation_type, item_name, feedback, now)
        )
        conn.commit()
