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





# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load the trained model
# model = joblib.load('models/eco_model.pkl')

# # ---------- Page Configuration ----------
# st.set_page_config(page_title="CO₂ Emission Estimator", page_icon="🌿", layout="centered")

# # ---------- Custom CSS for UI ----------
# st.markdown("""
#     <style>
#         body {
#             background-color: #f4f6f9;
#         }
#         .main {
#             background-color: white;
#             padding: 30px;
#             border-radius: 15px;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.1);
#         }
#         footer {
#             visibility: hidden;
#         }
#         .footer-text {
#             position: fixed;
#             bottom: 10px;
#             left: 0;
#             right: 0;
#             text-align: center;
#             color: gray;
#             font-size: 14px;
#         }
#     </style>
#     <div class="footer-text">
#         Developed with ❤️ by <b>Shetu</b>
#     </div>
# """, unsafe_allow_html=True)

# # ---------- Title ----------
# st.markdown("<h2 style='text-align: center; color: #2c3e50;'>🌍 CO₂ Emission Estimator</h2>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; color: #555;'>Predict your weekly carbon footprint based on lifestyle inputs</p>", unsafe_allow_html=True)
# st.markdown("---")

# # ---------- Input Fields ----------
# st.subheader("Enter the required details:")

# def get_user_input():
#     col1, col2 = st.columns(2)

#     with col1:
#         feature1 = st.number_input('Distance Travelled (km)', min_value=0.0, value=10.0)
#         feature2 = st.number_input('Electricity Used (kWh)', min_value=0.0, value=5.0)
    
#     with col2:
#         feature3 = st.number_input('Waste Generated (kg)', min_value=0.0, value=2.0)
#         feature4 = st.number_input('Water Usage (litres)', min_value=0.0, value=50.0)

#     data = {
#         'distance': feature1,
#         'electricity': feature2,
#         'waste': feature3,
#         'water': feature4
#     }

#     return pd.DataFrame([data])

# input_df = get_user_input()

# # ---------- Prediction ----------
# st.markdown("---")
# if st.button('🔍 Predict Emission'):
#     prediction = model.predict(input_df)[0]
#     st.success(f"🌱 Estimated Weekly CO₂ Emission: **{round(prediction, 2)} kg**")
