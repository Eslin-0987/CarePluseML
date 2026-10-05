import os
from app import create_app

# Instantiate the Flask application
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = app.config.get('DEBUG', False)
    app.run(host='0.0.0.0', port=port, debug=debug)
