from flask import Flask, request, render_template, jsonify
import numpy as np
from app_ml.model import model
from app_ml import app
import pandas as pd

# Define the labels map
labels_map = {
    0: "NO Heart Disease",
    1: "With Heart Disease",
}

# Load and preprocess data, create pipeline
model = model

@app.route("/")
@app.route("/index")
def index():
    print('index')
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    print('predict')
    try:
         # Get the form data
        form_data = request.form.to_dict()
        print('Received form data:', form_data)

        # Convert form data into a list of features
        features = [
            int(form_data['age']),
            1 if form_data['sex'] == 'male' else 0,
            {'typical_angina': 0, 'atypical_angina': 1, 'non_anginal_pain': 2, 'asymptomatic': 3}[form_data['chestPainType']],
            int(form_data['restingBP']),
            int(form_data['cholesterol']),
            int(form_data['fastingBS']),
            {'normal': 0, 'st_t_wave_abnormality': 1, 'left_ventricular_hypertrophy': 2}[form_data['restingECG']],
            int(form_data['maxHR']),
            1 if form_data['exerciseAngina'] == 'yes' else 0,
            float(form_data['oldpeak']),
            {'upsloping': 0, 'flat': 1, 'downsloping': 2}[form_data['st_slope']]
        ]
        print('Features before conversion:', features)

        # Convert features to numpy array and ensure they are in float32 format
        features = np.array(features, dtype=np.float32)
        print('Features after conversion to float32:', features)

        # Convert the numpy array to a DataFrame with correct column names
        feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        features_df = pd.DataFrame([features], columns=feature_names)
        print('Features DataFrame:', features_df)

        # Preprocess data and predict using the model
        prediction = model.predict(features_df)
        print('Prediction:', prediction)

        return render_template('result.html', predicted_class=labels_map[prediction[0]])

    except Exception as e:
        print('Error:', e)
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)