from flask import Blueprint, render_template, current_app, g
from app.routes.auth import auth_required
from app.database.db import (
    get_user_dashboard_stats, get_health_profile, get_user_predictions, get_user_activity
)

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard')
@auth_required
def dashboard():
    """
    Main user dashboard presenting real calculated metrics, health summary,
    recent predictions, and recent activities.
    """
    db_path = current_app.config['DATABASE_PATH']
    user_id = g.user['id']
    
    # 1. Calculated real statistics from SQLite
    stats = get_user_dashboard_stats(db_path, user_id)
    
    # 2. User health profile
    health_profile = get_health_profile(db_path, user_id)
    
    # 3. Recent 5 predictions for this user
    recent_predictions = get_user_predictions(db_path, user_id, limit=5)
    
    # 4. Recent activities
    recent_activities = get_user_activity(db_path, user_id, limit=5)
    
    # Determine personalization state
    is_new_user = (stats['total_predictions'] == 0)
    
    return render_template(
        'dashboard.html',
        user=g.user,
        stats=stats,
        health_profile=health_profile,
        recent_predictions=recent_predictions,
        recent_activities=recent_activities,
        is_new_user=is_new_user,
        disclaimer=current_app.config['DISCLAIMER']
    )
