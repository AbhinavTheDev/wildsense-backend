import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from functools import wraps 
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# --- Authentication Configuration ---
load_dotenv()
API_KEY = os.getenv("API_KEY")

# --- Configuration ---
MODEL_DIR = 'models'
LIFESPAN_MODEL_PATH = os.path.join(MODEL_DIR, 'lifespan_model.pkl')
LIFESPAN_SCALER_PATH = os.path.join(MODEL_DIR, 'lifespan_scaler.pkl')
LIFESPAN_NUM_COLS_PATH = os.path.join(MODEL_DIR, 'lifespan_numerical_cols.pkl')
LIFESPAN_ALL_FEATURES_PATH = os.path.join(MODEL_DIR, 'lifespan_all_features.pkl')

CONSERVATION_MODEL_PATH = os.path.join(MODEL_DIR, 'conservation_model.pkl')
CONSERVATION_SCALER_PATH = os.path.join(MODEL_DIR, 'conservation_scaler.pkl')
CONSERVATION_FEATURES_PATH = os.path.join(MODEL_DIR, 'conservation_features.pkl')

# --- Load Models and Scalers ---
try:
    lifespan_model = joblib.load(LIFESPAN_MODEL_PATH)
    lifespan_scaler = joblib.load(LIFESPAN_SCALER_PATH)
    lifespan_numerical_cols = joblib.load(LIFESPAN_NUM_COLS_PATH)
    lifespan_all_features = joblib.load(LIFESPAN_ALL_FEATURES_PATH)

    conservation_model = joblib.load(CONSERVATION_MODEL_PATH)
    conservation_scaler = joblib.load(CONSERVATION_SCALER_PATH)
    conservation_features = joblib.load(CONSERVATION_FEATURES_PATH)

    print("Models and scalers loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model/scaler: {e}")
    print("Ensure model files exist in the 'models' directory after running training scripts.")
    # Exit or handle appropriately if models can't be loaded
    lifespan_model = None
    conservation_model = None
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    lifespan_model = None
    conservation_model = None

# --- Helper Function for Error Response ---
def make_error_response(message, status_code):
    return jsonify({"error": message}), status_code

# --- Authentication Decorator ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the API key from the request header
        api_key = request.headers.get('X-API-Key')
        
        # Check if key is valid
        if not api_key or api_key != API_KEY:
            return jsonify({"error": "Unauthorized - Valid API key required"}), 401
        
        # If key is valid, proceed to the route function
        return f(*args, **kwargs)
    
    return decorated_function

# --- Routes ---
@app.route('/')
def home():
    return "Wildsense Model API is live."

@app.route('/predict', methods=['GET'])
@require_api_key
def predict():
    return make_error_response("Please specify a prediction type (lifespan or conservation) in the URL.", 400)

@app.route('/predict/lifespan', methods=['POST'])
@require_api_key
def predict_lifespan():
    if not lifespan_model or not lifespan_scaler:
        return make_error_response("Lifespan model not loaded.", 500)

    if not request.is_json:
        return make_error_response("Request must be JSON.", 400)

    data = request.get_json()

    # Basic Input Validation (check for presence of keys)
    required_keys = lifespan_all_features # Use the loaded feature list
    if not all(key in data for key in required_keys):
        missing = [key for key in required_keys if key not in data]
        return make_error_response(f"Missing required keys: {', '.join(missing)}", 400)

    try:
        input_df = pd.DataFrame([data], columns=lifespan_all_features)

        # Separate numerical and non-numerical features
        numerical_df = input_df[lifespan_numerical_cols]
        non_numerical_df = input_df.drop(columns=lifespan_numerical_cols)

        # Scale numerical features using the loaded scaler
        # IMPORTANT: Use transform, NOT fit_transform
        numerical_scaled = lifespan_scaler.transform(numerical_df)
        numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=lifespan_numerical_cols, index=input_df.index)

        # Recombine features in the correct order for the model
        # This order must exactly match how the model was trained
        # The lifespan_all_features list defines this order.
        # Assuming non-numerical first, then scaled numerical:
        final_input_df = pd.concat([non_numerical_df, numerical_scaled_df], axis=1)[lifespan_all_features]

        # Make prediction
        prediction = lifespan_model.predict(final_input_df)

        # Return prediction as JSON
        predicted_value = int(round(prediction[0])) # Convert to int after rounding

        return jsonify({"predicted_lifespan": predicted_value})

    except ValueError as e:
        return make_error_response(f"Invalid input data type or value: {e}", 400)
    except Exception as e:
        print(f"Error during lifespan prediction: {e}") # Log the error server-side
        return make_error_response("An error occurred during prediction.", 500)


@app.route('/predict/conservation', methods=['POST'])
@require_api_key
def predict_conservation():
    if not conservation_model or not conservation_scaler:
        return make_error_response("Conservation model not loaded.", 500)

    if not request.is_json:
        return make_error_response("Request must be JSON.", 400)

    data = request.get_json()

    # Basic Input Validation
    required_keys = conservation_features # Use the loaded feature list
    if not all(key in data for key in required_keys):
         missing = [key for key in required_keys if key not in data]
         return make_error_response(f"Missing required keys: {', '.join(missing)}", 400)

    try:
        input_df = pd.DataFrame([data], columns=conservation_features)

        # Scale features using the loaded scaler
        input_scaled = conservation_scaler.transform(input_df)

        # --- Prediction Logic ---
        # Get probabilities for class 1 (Endangered)
        predict_probs = conservation_model.predict_proba(input_scaled)[:, 1]

        # Apply threshold (0.25 in your script) to classify
        threshold = 0.25
        prediction = (predict_probs >= threshold).astype(int)
        # --- End Prediction Logic ---

        # Return prediction as JSON
        predicted_status = int(prediction[0]) # Get the first element (0 or 1)

        return jsonify({"predicted_conservation_status": predicted_status,
                        "probability_endangered": float(predict_probs[0]) # Optional: return probability
                       })

    except ValueError as e:
         return make_error_response(f"Invalid input data type or value: {e}", 400)
    except Exception as e:
        print(f"Error during conservation prediction: {e}") # Log the error server-side
        return make_error_response("An error occurred during prediction.", 500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)