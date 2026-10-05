import os
import logging
from datetime import datetime
from flask import Flask, redirect, url_for, render_template, g, request
from flask_jwt_extended import JWTManager, verify_jwt_in_request, get_jwt_identity
from config import Config
from app.database.db import init_db, get_user_by_id
from app.services.prediction_service import prediction_service
from app.services.recommendation_service import recommendation_service

# Blueprints
from app.routes.auth import auth_bp
from app.routes.dashboard import dashboard_bp
from app.routes.prediction import prediction_bp
from app.routes.recommendation import recommendation_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    """Application factory for CarePulse Flask application."""
    # Note: templates and static folders are located at the project root
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    template_dir = os.path.join(basedir, 'templates')
    static_dir = os.path.join(basedir, 'static')

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config.from_object(config_class)

    # 1. Initialize SQLite Database
    try:
        init_db(app.config['DATABASE_PATH'])
        logger.info(f"Database verified at: {app.config['DATABASE_PATH']}")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

    # 2. Initialize Flask-JWT-Extended
    jwt = JWTManager(app)

    @jwt.unauthorized_loader
    def unauthorized_callback(callback):
        return redirect(url_for('auth.login', next=request.path))

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return redirect(url_for('auth.login', next=request.path))

    # 3. Initialize ML Models & Recommendation Datasets
    try:
        prediction_service.load_models(app.config['MODELS_DIR'])
        logger.info("Machine learning prediction service initialized.")
    except Exception as e:
        logger.error(f"ML Model Loading failed: {e}")

    try:
        recommendation_service.load_data(app.config['DATA_RECOMMENDATION_DIR'])
        logger.info("Recommendation knowledge service initialized.")
    except Exception as e:
        logger.error(f"Recommendation service loading failed: {e}")

    # 4. Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(recommendation_bp)

    # 5. Global Template Filters & Context
    @app.template_filter('format_date')
    def format_date_filter(value):
        """Format ISO / SQLite timestamps into clean, human-readable date strings."""
        if not value:
            return "N/A"
        try:
            dt = datetime.strptime(str(value).split('.')[0], '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%d %B %Y')
        except Exception:
            try:
                dt = datetime.fromisoformat(str(value))
                return dt.strftime('%d %B %Y')
            except Exception:
                return str(value)

    @app.before_request
    def load_authenticated_user():
        """Optionally populate g.user if a valid JWT token is present in request cookies."""
        g.user = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
            if user_id:
                g.user = get_user_by_id(app.config['DATABASE_PATH'], int(user_id))
        except Exception:
            g.user = None

    @app.context_processor
    def inject_global_variables():
        """Make application metadata and user globally available to templates."""
        return {
            'app_name': app.config['APP_NAME'],
            'current_user': getattr(g, 'user', None),
            'disclaimer': app.config['DISCLAIMER'],
            'current_year': datetime.now().year
        }

    # 6. Root Route
    @app.route('/')
    def index():
        """Root route: Redirect to dashboard if authenticated, otherwise to login."""
        if getattr(g, 'user', None):
            return redirect(url_for('dashboard.dashboard'))
        return redirect(url_for('auth.login'))

    # 7. Global Error Handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('about.html', error_title="Page Not Found (404)", error_msg="The page you requested could not be located."), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error: {e}")
        return render_template('about.html', error_title="System Notice (500)", error_msg="An unexpected error occurred while processing your request. Please try again shortly."), 500

    return app
