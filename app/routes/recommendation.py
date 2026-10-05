import re
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, current_app, g, jsonify
)
from app.routes.auth import auth_required
from app.database.db import (
    get_health_profile, upsert_health_profile, update_user_account,
    add_feedback, log_activity
)

recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route('/profile', methods=['GET', 'POST'])
@auth_required
def profile():
    """
    User Profile Management:
    Clearly distinguishes Account Information (Name, Email, Created Date)
    from Clinical Health Profile (Age, Gender, Vitals, Medical History).
    """
    db_path = current_app.config['DATABASE_PATH']
    user_id = g.user['id']
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        age_str = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        blood_pressure = request.form.get('blood_pressure', '').strip()
        glucose_str = request.form.get('glucose', '').strip()
        heart_rate_str = request.form.get('heart_rate', '').strip()
        cholesterol_str = request.form.get('cholesterol', '').strip()
        medical_history = request.form.get('medical_history', '').strip()
        
        errors = []
        if not name or len(name) < 2:
            errors.append("Name must be at least 2 characters.")
            
        age = None
        if age_str:
            try:
                age = int(age_str)
                if age < 1 or age > 125:
                    errors.append("Age must be between 1 and 125.")
            except ValueError:
                errors.append("Age must be a valid number.")
                
        glucose = None
        if glucose_str:
            try:
                glucose = float(glucose_str)
                if glucose < 20 or glucose > 600:
                    errors.append("Fasting glucose value appears out of realistic physiological range (20-600 mg/dL).")
            except ValueError:
                errors.append("Glucose must be a valid number.")
                
        heart_rate = None
        if heart_rate_str:
            try:
                heart_rate = int(heart_rate_str)
                if heart_rate < 30 or heart_rate > 250:
                    errors.append("Heart rate appears out of realistic range (30-250 bpm).")
            except ValueError:
                errors.append("Heart rate must be an integer.")
                
        cholesterol = None
        if cholesterol_str:
            try:
                cholesterol = float(cholesterol_str)
                if cholesterol < 50 or cholesterol > 800:
                    errors.append("Cholesterol value appears out of realistic range (50-800 mg/dL).")
            except ValueError:
                errors.append("Cholesterol must be a valid number.")

        if errors:
            for err in errors:
                flash(err, "error")
            health_profile = get_health_profile(db_path, user_id)
            return render_template('profile.html', user=g.user, health_profile=health_profile), 400

        # Update User identity
        update_user_account(db_path, user_id, name, age=age, gender=gender)
        
        # Update Health Profile
        upsert_health_profile(
            db_path=db_path,
            user_id=user_id,
            age=age,
            gender=gender,
            blood_pressure=blood_pressure or None,
            glucose=glucose,
            heart_rate=heart_rate,
            cholesterol=cholesterol,
            medical_history=medical_history or None
        )
        
        log_activity(db_path, user_id, "Profile Updated", "User updated health profile and account information.")
        flash("Profile updated successfully.", "success")
        return redirect(url_for('recommendation.profile'))
        
    health_profile = get_health_profile(db_path, user_id)
    return render_template(
        'profile.html',
        user=g.user,
        health_profile=health_profile,
        disclaimer=current_app.config['DISCLAIMER']
    )

@recommendation_bp.route('/about')
def about():
    """
    Educational Methodology & Machine Learning Architecture page.
    Explains the dataset, Random Forest vs Linear SVC, knowledge-based recommendations,
    system limitations, and ethical healthcare disclaimers.
    """
    user = getattr(g, 'user', None)
    return render_template(
        'about.html',
        user=user,
        disclaimer=current_app.config['DISCLAIMER']
    )

@recommendation_bp.route('/api/feedback', methods=['POST'])
@auth_required
def submit_feedback():
    """Submit user feedback on recommendation utility for research purposes."""
    data = request.get_json(silent=True) or request.form
    rec_type = data.get('recommendation_type', '').strip()
    item_name = data.get('item_name', '').strip()
    feedback = data.get('feedback', '').strip()
    
    if not feedback:
        return jsonify({"success": False, "error": "Feedback content is required."}), 400
        
    db_path = current_app.config['DATABASE_PATH']
    add_feedback(db_path, g.user['id'], rec_type, item_name, feedback)
    return jsonify({"success": True, "message": "Feedback recorded. Thank you for contributing to research!"})
