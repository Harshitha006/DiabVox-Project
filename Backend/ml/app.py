from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the ML model and scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
lr = pickle.load(open('lr.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received JSON Data:", data)  # Debugging Line

        input_data = [
            data["Pregnancies"], data["Glucose"], data["BloodPressure"], 
            data["SkinThickness"], data["Insulin"], data["BMI"], 
            data["DiabetesPedigreeFunction"], data["Age"]
        ]

        print("Processed Input Data:", input_data)  # Debugging Line

        scaled_data = scaler.transform([input_data])
        prediction = lr.predict(scaled_data)[0]

        return jsonify({"prediction": "indicates" if prediction == 1 else "does not indicate"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

