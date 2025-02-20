const express = require('express');
const WaterLog = require('../models/WaterLog');
const router = express.Router();

// Log Water Intake
router.post('/log', async (req, res) => {
    try {
        const newLog = new WaterLog(req.body);
        await newLog.save();
        res.json({ message: '✅ Water intake logged' });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// Get Water Logs for a User
router.get('/:userId', async (req, res) => {
    try {
        const logs = await WaterLog.find({ userId: req.params.userId });
        res.json(logs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
