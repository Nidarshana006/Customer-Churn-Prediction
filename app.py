from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np

app = Flask(__name__)
CORS(app)

# Model load
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature names load
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [data.get(feature, 0) for feature in feature_names]
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    
    return jsonify({
        'churn': int(prediction),
        'probability': round(float(probability) * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)