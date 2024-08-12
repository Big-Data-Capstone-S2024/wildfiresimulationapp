from flask import Flask, jsonify
from modules.historical_data import historical_data_bp  # Ensure this import is correct
from modules.user import user_bp  # Assuming you have other blueprints
from modules.model_predict import model_predict_bp  # Assuming you have other blueprints
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(historical_data_bp, url_prefix='/api')
app.register_blueprint(user_bp, url_prefix='/api')  # Assuming you have other blueprints
app.register_blueprint(model_predict_bp, url_prefix='/api')  # Assuming you have other blueprints

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Wildfire Simulation API"}), 200

if __name__ == '__main__':
    app.run(debug=True)
