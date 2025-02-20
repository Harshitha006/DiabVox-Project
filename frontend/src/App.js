// ---- Frontend ----
// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
    const [healthData, setHealthData] = useState([]);
    const [dietPlan, setDietPlan] = useState(null);
    const [voiceCommand, setVoiceCommand] = useState('');
    const [aiPrediction, setAiPrediction] = useState(null);

    useEffect(() => {
        const userId = localStorage.getItem('userId');
        if (!userId) return;

        axios.get(`http://localhost:5000/api/health/${userId}`)
            .then(res => setHealthData(res.data))
            .catch(err => console.log(err));

        axios.get(`http://localhost:5000/api/diet/${userId}`)
            .then(res => setDietPlan(res.data))
            .catch(err => console.log(err));
    }, [healthData]);

    const checkDiabetesRisk = async () => {
        const latestLog = healthData[healthData.length - 1];
        if (!latestLog) return alert('No health data available');
        
        const response = await axios.post('http://localhost:5000/api/ai/predict', {
            glucoseLevel: latestLog.glucoseLevel,
            insulinLevel: latestLog.insulinLevel
        });
        setAiPrediction(response.data);
    };

    return (
        <div>
            <h1>Health Tracker</h1>
            {healthData.map(log => (
                <div key={log._id}>
                    <p>Date: {new Date(log.date).toDateString()}</p>
                    <p>Calories: {log.calories}</p>
                    <p>Water: {log.water} L</p>
                    <p>Sleep: {log.sleep} hrs</p>
                    <p>Glucose Level: {log.glucoseLevel} mg/dL</p>
                    <p>Insulin Level: {log.insulinLevel} μU/mL</p>
                </div>
            ))}
            <button onClick={checkDiabetesRisk}>Check Diabetes Risk</button>
            {aiPrediction && <p>Risk: {aiPrediction.message} (Score: {aiPrediction.riskScore})</p>}
            <h2>Diet Plan</h2>
            {dietPlan ? (
                <div>
                    <h3>Recommended Foods</h3>
                    <ul>{dietPlan.recommendedFoods.map(food => <li key={food}>{food}</li>)}</ul>
                    <h3>Restricted Foods</h3>
                    <ul>{dietPlan.restrictedFoods.map(food => <li key={food}>{food}</li>)}</ul>
                </div>
            ) : <p>No diet plan available.</p>}
        </div>
    );
}
export default App;
