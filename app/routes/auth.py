from functools import wraps
import re
from datetime import datetime
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, current_app, make_response, g
)
from flask_jwt_extended import (
    create_access_token, set_access_cookies, unset_jwt_cookies,
    verify_jwt_in_request, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
from app.database.db import (
    create_user, get_user_by_email, get_user_by_id, log_activity
)

auth_bp = Blueprint('auth', __name__)

EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

def auth_required(f):
    """
    Decorator to protect web routes.
    Verifies JWT token or Flask signed session. If invalid or expired,
    clears stale credentials and redirects to login with an informative flash message.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import session
        user = None
        user_id = session.get('user_id')
        
        if not user_id:
            try:
                verify_jwt_in_request(optional=True)
                user_id = get_jwt_identity()
            except Exception:
                user_id = None
                
        if user_id:
            db_path = current_app.config['DATABASE_PATH']
            try:
                user = get_user_by_id(db_path, int(user_id))
            except Exception:
                import time
                time.sleep(0.05)
                try:
                    user = get_user_by_id(db_path, int(user_id))
                except Exception:
                    user = None
            
        if user:
            g.user = user
            session['user_id'] = user['id']
            return f(*args, **kwargs)

        # Token / session missing, invalid, or expired
        response = make_response(redirect(url_for('auth.login', next=request.path)))
        unset_jwt_cookies(response)
        session.clear()
        if user_id:
            flash("Your session has expired. Please log in to continue.", "warning")
        else:
            flash("Please log in to access this page.", "info")
        return response
            
    return decorated_function

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration flow with server-side validation and password hashing."""
    from flask import session
    # If already logged in, do not let another user accidentally overwrite the active session
    if request.method == 'GET' and (getattr(g, 'user', None) or session.get('user_id')):
        flash("You are already signed in. Please Sign Out first if you wish to register a different account.", "info")
        return redirect(url_for('dashboard.view_dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        age_str = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        
        errors = []
        
        if not name or len(name) < 2:
            errors.append("Please provide your full name (minimum 2 characters).")
            
        if not email or not re.match(EMAIL_REGEX, email):
            errors.append("Please provide a valid email address.")
            
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters in length.")
            
        if password != confirm_password:
            errors.append("Password confirmation does not match.")
            
        age = None
        if age_str:
            try:
                age = int(age_str)
                if age < 1 or age > 125:
                    errors.append("Please enter a realistic age between 1 and 125.")
            except ValueError:
                errors.append("Age must be a valid whole number.")
                
        allowed_genders = ['Male', 'Female', 'Non-Binary', 'Other', 'Prefer not to say']
        if gender and gender not in allowed_genders:
            errors.append("Please select a valid gender option.")

        db_path = current_app.config['DATABASE_PATH']
        
        # Check if email is already registered
        if not errors:
            existing = get_user_by_email(db_path, email)
            if existing:
                errors.append("An account with this email address already exists. Please log in.")
                
        if errors:
            for err in errors:
                flash(err, "error")
            return render_template('signup.html', 
                                   name=name, email=email, age=age_str, gender=gender), 400
                                   
        # Secure password hashing
        password_hash = generate_password_hash(password, method='scrypt')
        
        try:
            user_id = create_user(db_path, name, email, password_hash, age=age, gender=gender)
            log_activity(db_path, user_id, "User Account Created", "Completed registration.")
            
            # Store in Flask session & generate JWT token
            session['user_id'] = user_id
            session['user_name'] = name
            session.permanent = True
            
            access_token = create_access_token(identity=str(user_id))
            
            flash("Account created successfully! Welcome to HealthRecom Decision Support.", "success")
            response = make_response(redirect(url_for('dashboard.view_dashboard')))
            set_access_cookies(response, access_token)
            return response
            
        except Exception as e:
            current_app.logger.error(f"Error during registration: {e}")
            flash("An unexpected error occurred while creating your account. Please try again.", "error")
            return render_template('signup.html', name=name, email=email, age=age_str, gender=gender), 500
            
    return render_template('signup.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login flow with password verification and JWT token issuance."""
    from flask import session
    # If already logged in, redirect to dashboard rather than re-displaying login
    if request.method == 'GET' and (getattr(g, 'user', None) or session.get('user_id')):
        flash("You are already logged in.", "info")
        return redirect(url_for('dashboard.view_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        next_url = request.args.get('next') or request.form.get('next')
        
        if not email or not password:
            flash("Please provide both email and password.", "error")
            return render_template('login.html', email=email), 400
            
        db_path = current_app.config['DATABASE_PATH']
        user = get_user_by_email(db_path, email)
        
        if not user or not check_password_hash(user['password_hash'], password):
            flash("Invalid email or password. Please try again.", "error")
            return render_template('login.html', email=email), 401
            
        # Success: store in session & generate JWT
        user_id = user['id']
        session['user_id'] = user_id
        session['user_name'] = user['name']
        session.permanent = True
        
        access_token = create_access_token(identity=str(user_id))
        log_activity(db_path, user_id, "User Logged In", "Successful authentication.")
        
        flash(f"Welcome back, {user['name']}!", "success")
        
        # Validate next_url for safe internal redirects
        target = next_url if next_url and next_url.startswith('/') else url_for('dashboard.view_dashboard')
        response = make_response(redirect(target))
        set_access_cookies(response, access_token)
        return response
        
    return render_template('login.html', next=request.args.get('next'))

@auth_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    """Invalidate authentication state and redirect to login."""
    from flask import session
    try:
        user_id = session.get('user_id')
        if not user_id:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        if user_id:
            db_path = current_app.config['DATABASE_PATH']
            log_activity(db_path, int(user_id), "User Logged Out", "Session ended.")
    except Exception:
        pass
        
    session.clear()
    response = make_response(redirect(url_for('auth.login')))
    unset_jwt_cookies(response)
    flash("You have been securely logged out.", "info")
    return response

