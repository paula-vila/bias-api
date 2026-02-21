from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load('bias_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return jsonify({'bias': prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
