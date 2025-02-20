// ---- Routes ----
// routes/aiRoutes.js
const express = require('express');
const router = express.Router();

router.post('/predict', async (req, res) => {
    try {
        const { glucoseLevel, insulinLevel } = req.body;
        const riskScore = (glucoseLevel * 0.6) + (insulinLevel * 0.4);
        res.json({ riskScore, message: riskScore > 150 ? 'High Risk' : 'Low Risk' });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

module.exports = router;