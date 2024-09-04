from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/hearing_test_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = int(request.form['age'])
    physical_score = float(request.form['physical_score'])

    # Prepare the input data as a NumPy array
    input_data = np.array([[age, physical_score]])

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Map the prediction to a human-readable label
    prediction_label = "Good" if prediction == 1 else "Bad"

    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
