from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    model = joblib.load('bias_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    
    scores = model.decision_function(vec)[0]
    scores = np.array(scores)
    sorted_scores = np.sort(scores)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]
    confidence = round(min(float(margin) / 2.0 * 100, 100))

    scores = model.decision_function(vec)[0]
    scores = np.array(scores)
    sorted_scores = np.sort(scores)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]
    
    # Temporary debug - remove later
    print(f"Margin: {margin}, Raw scores: {sorted_scores[:3]}")
    
    confidence = round(min(float(margin) / 2.0 * 100, 100))
    
    return jsonify({'bias': prediction, 'confidence': confidence, 'debug_margin': float(margin)})

    return jsonify({'bias': prediction, 'confidence': confidence})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
