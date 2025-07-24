from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('models/eco_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    
    # Convert values to float and reshape for prediction
    features = np.array([float(value) for value in data.values()]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]

    # Return result to the webpage
    return render_template('index.html', prediction_text=f'Estimated Weekly CO₂ Emission: {round(prediction, 2)} kg')

if __name__ == '__main__':
    app.run(debug=True)