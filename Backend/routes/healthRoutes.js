const express = require('express');
const HealthLog = require('../models/HealthLog');
const router = express.Router();

// Log Health Data
router.post('/log', async (req, res) => {
    try {
        const newLog = new HealthLog(req.body);
        await newLog.save();
        res.json({ message: '✅ Health data logged' });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// Get Health Logs for a User
router.get('/:userId', async (req, res) => {
    try {
        const logs = await HealthLog.find({ userId: req.params.userId });
        res.json(logs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
