from flask import Flask, jsonify, send_from_directory
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='build')

# Access the environment variables
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')
supabase_headers = {
    "apikey": supabase_key,
    "Authorization": f"Bearer {supabase_key}",
    "Content-Type": "application/json"
}

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Wildfire Simulation API"}), 200

@app.route('/api/users')
def get_users():
    try:
        # Making a GET request to Supabase API to list users
        response = requests.get(f"{supabase_url}/auth/v1/users", headers=supabase_headers)
        response.raise_for_status()  # Raises an error for bad responses
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
