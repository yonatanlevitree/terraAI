import os
from flask import Flask, redirect
from flask_cors import CORS
from functions.v1 import simulate

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Register routes from job type modules
app.register_blueprint(simulate.simulate_bp)

@app.route('/')
def root():
    """Redirect to API documentation"""
    return redirect('/swagger')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "firebase" if os.getenv('FIREBASE', 'False').lower() == 'true' else "in-memory"
    }

if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0', port=5001) 