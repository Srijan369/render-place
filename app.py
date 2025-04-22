from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_score = float(request.form.get('profile_score'))

        # Prepare input for prediction
        input_features = np.array([[cgpa, iq, profile_score]])
        prediction = model.predict(input_features)[0]

        result = 'Placed' if prediction == 1 else 'Not Placed'
        return jsonify({'placement': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
