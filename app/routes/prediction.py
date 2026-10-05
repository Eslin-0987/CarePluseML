import json
import logging
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash, current_app, g, abort
)
from app.routes.auth import auth_required
from app.services.preprocessing import get_categorized_symptoms, get_all_symptoms_flat, clean_feature_label
from app.services.prediction_service import prediction_service
from app.services.recommendation_service import recommendation_service
from app.database.db import (
    get_health_profile, upsert_health_profile, create_prediction,
    get_user_predictions, get_prediction_by_id, log_activity
)

logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/assessment', methods=['GET', 'POST'])
@auth_required
def assessment():
    """
    Health assessment and symptom submission page.
    Generates prediction using actual trained Random Forest or Linear SVC models.
    """
    db_path = current_app.config['DATABASE_PATH']
    user_id = g.user['id']
    
    if request.method == 'POST':
        # 1. Extract and validate model selection
        model_choice = request.form.get('model_selection', '').strip().lower()
        if model_choice not in ('random_forest', 'linear_svc'):
            flash("Please choose a valid machine-learning model (Random Forest or Linear SVC).", "error")
            return redirect(url_for('prediction.assessment'))
            
        # 2. Extract selected symptoms
        selected_symptoms = request.form.getlist('symptoms')
        # Also support JSON array or comma-separated if submitted via JS
        if not selected_symptoms:
            symptoms_raw = request.form.get('symptoms_payload', '')
            if symptoms_raw:
                try:
                    selected_symptoms = json.loads(symptoms_raw)
                except Exception:
                    selected_symptoms = [s.strip() for s in symptoms_raw.split(',') if s.strip()]
                    
        if not selected_symptoms:
            flash("Please select at least one current symptom to proceed with the assessment.", "error")
            return redirect(url_for('prediction.assessment'))
            
        # 3. Extract clinical / personal measurements
        age_str = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()
        blood_pressure = request.form.get('blood_pressure', '').strip()
        glucose_str = request.form.get('glucose', '').strip()
        heart_rate_str = request.form.get('heart_rate', '').strip()
        cholesterol_str = request.form.get('cholesterol', '').strip()
        medical_history = request.form.get('medical_history', '').strip()
        
        # Parse numeric vitals safely
        age = int(age_str) if age_str.isdigit() else g.user.get('age')
        glucose = float(glucose_str) if glucose_str else None
        heart_rate = int(heart_rate_str) if heart_rate_str.isdigit() else None
        cholesterol = float(cholesterol_str) if cholesterol_str else None
        
        # 4. Execute ML Prediction Pipeline
        try:
            prediction_result = prediction_service.predict(model_choice, selected_symptoms)
        except Exception as e:
            logger.error(f"Prediction execution failed: {e}")
            flash(
                "We could not generate the prediction because the assessment data could not be processed. "
                "Please review your information and try again.", 
                "error"
            )
            return redirect(url_for('prediction.assessment'))

        # 5. Format symptom string for database record
        symptoms_serialized = ", ".join(prediction_result['recognized_symptoms'][i]['id'] for i in range(len(prediction_result['recognized_symptoms'])))
        
        # 6. Save prediction in database
        try:
            pred_id = create_prediction(
                db_path=db_path,
                user_id=user_id,
                age=age,
                gender=gender or g.user.get('gender'),
                symptoms=symptoms_serialized,
                blood_pressure=blood_pressure or None,
                glucose=glucose,
                heart_rate=heart_rate,
                cholesterol=cholesterol,
                medical_history=medical_history or None,
                model_used=prediction_result['model_name'],
                predicted_disease=prediction_result['predicted_disease'],
                prediction_probability=prediction_result['probability_estimate']
            )
            
            # 7. Update health profile with latest vitals if provided
            upsert_health_profile(
                db_path=db_path,
                user_id=user_id,
                age=age,
                gender=gender or g.user.get('gender'),
                blood_pressure=blood_pressure or None,
                glucose=glucose,
                heart_rate=heart_rate,
                cholesterol=cholesterol,
                medical_history=medical_history or None
            )
            
            log_activity(
                db_path, user_id, "Assessment Completed",
                f"Generated prediction using {prediction_result['model_name']}: {prediction_result['predicted_disease']}."
            )
            
            flash("Assessment successfully analyzed! Here is your prediction result.", "success")
            return redirect(url_for('prediction.result', prediction_id=pred_id))
            
        except Exception as e:
            logger.error(f"Error saving prediction to database: {e}")
            flash("Prediction was computed but an error occurred while saving to your history.", "error")
            return redirect(url_for('dashboard.dashboard'))

    # GET Request: Prepare pre-filled profile and symptom catalog
    health_profile = get_health_profile(db_path, user_id)
    categorized_symptoms = get_categorized_symptoms()
    flat_symptoms = get_all_symptoms_flat()
    
    log_activity(db_path, user_id, "Assessment Started", "Opened health assessment form.")
    
    return render_template(
        'assessment.html',
        user=g.user,
        health_profile=health_profile,
        categorized_symptoms=categorized_symptoms,
        flat_symptoms_json=json.dumps(flat_symptoms),
        disclaimer=current_app.config['DISCLAIMER']
    )

@prediction_bp.route('/result/<int:prediction_id>')
@auth_required
def result(prediction_id):
    """
    Display prediction results and corresponding educational recommendations.
    Strictly isolated: users can only view their own predictions.
    """
    db_path = current_app.config['DATABASE_PATH']
    user_id = g.user['id']
    
    prediction = get_prediction_by_id(db_path, prediction_id, user_id)
    if not prediction:
        flash("The requested prediction record was not found or access is unauthorized.", "warning")
        return redirect(url_for('dashboard.dashboard'))
        
    # Format symptoms for display
    symptom_ids = [s.strip() for s in (prediction['symptoms'] or '').split(',') if s.strip()]
    formatted_symptoms = [clean_feature_label(s) for s in symptom_ids]
    
    # Retrieve educational recommendations from knowledge base
    recommendations = recommendation_service.get_recommendations(prediction['predicted_disease'])
    
    log_activity(db_path, user_id, "Recommendation Viewed", f"Inspected details for {prediction['predicted_disease']}.")
    
    return render_template(
        'result.html',
        user=g.user,
        prediction=prediction,
        formatted_symptoms=formatted_symptoms,
        recommendations=recommendations,
        disclaimer=current_app.config['DISCLAIMER']
    )

@prediction_bp.route('/history')
@auth_required
def history():
    """
    Prediction history page with search, model filtering, and sorting.
    Strictly user-isolated.
    """
    db_path = current_app.config['DATABASE_PATH']
    user_id = g.user['id']
    
    search_query = request.args.get('q', '').strip()
    model_filter = request.args.get('model', 'all').strip()
    sort_order = request.args.get('sort', 'desc').strip()
    
    predictions = get_user_predictions(
        db_path=db_path,
        user_id=user_id,
        search=search_query if search_query else None,
        model_filter=model_filter if model_filter in ('Random Forest', 'Linear SVC') else None,
        sort_order=sort_order
    )
    
    # Add formatted symptoms count and label
    for p in predictions:
        sym_list = [s.strip() for s in (p['symptoms'] or '').split(',') if s.strip()]
        p['symptom_count'] = len(sym_list)
        p['formatted_symptoms_preview'] = ", ".join(clean_feature_label(s) for s in sym_list[:3])
        if len(sym_list) > 3:
            p['formatted_symptoms_preview'] += f" +{len(sym_list) - 3} more"
            
    log_activity(db_path, user_id, "History Viewed", "Viewed assessment history records.")
    
    return render_template(
        'history.html',
        user=g.user,
        predictions=predictions,
        search_query=search_query,
        model_filter=model_filter,
        sort_order=sort_order,
        disclaimer=current_app.config['DISCLAIMER']
    )
