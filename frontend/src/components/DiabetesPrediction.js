import React, { useState } from "react";
import axios from "axios";

function DiabetesPrediction() {
  const [formData, setFormData] = useState({
    Age: "", Glucose: "", BloodPressure: "", Insulin: "", 
    BMI: "", SkinThickness: "", DiabetesPedigreeFunction: ""
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await axios.post("http://localhost:5000/api/ai/predict", formData);
    setResult(response.data.prediction);
  };

  return (
    <div>
      <h1>Diabetes Prediction</h1>
      <form onSubmit={handleSubmit}>
        {Object.keys(formData).map((key) => (
          <input key={key} name={key} value={formData[key]} onChange={handleChange} placeholder={key} required />
        ))}
        <button type="submit">Predict</button>
      </form>
      {result && <h2>Prediction: {result}</h2>}
    </div>
  );
}

export default DiabetesPrediction;
