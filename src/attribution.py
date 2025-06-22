from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgboost_model.joblib')
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert input data to numpy array
        input_data = np.array(data['features'])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 