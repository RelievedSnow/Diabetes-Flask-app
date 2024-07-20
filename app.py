from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("index.html", href='static/diabetic_bg.jpg', avatar='static/avatar.jpg')
    else:
        # Get form data
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']

        # Convert inputs to numpy array
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        input_data = np.array(input_data, dtype=float).reshape(1, -1)

        # Load the model and scaler
        model = load('classifier.joblib')
        scaler = load('scaler.joblib')

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Return the result
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        return render_template("result.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
