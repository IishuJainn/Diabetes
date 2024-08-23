from flask import Flask, render_template, request
import numpy as np
import pickle
# from tensorflow import keras

app = Flask(__name__)
# "https://www.kaggle.com/code/ishujain25/diabetes-prediction/notebook"
with open('svm_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# model = keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template with the input form

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = float(request.form['age'])

    # Create a numpy array with the input values
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Scale the input data using the same StandardScaler object used to train the SVM model
    with open('scaler.pkl', 'rb') as fil:
        scaler = pickle.load(fil)
    standardized_input_data = scaler.transform(input_data)

    # Use the SVM model to predict the outcome (0 or 1)
    prediction = model.predict(standardized_input_data)
    if prediction == 1:
        pred="Diabetic"
    else:
        pred="Not Diabetic"

    # Render the HTML template with the prediction
    return render_template('result.html', prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)
