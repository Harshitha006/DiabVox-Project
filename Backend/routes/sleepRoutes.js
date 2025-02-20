const express = require('express');
const SleepLog = require('../models/SleepLog');
const router = express.Router();

// Log Sleep Data
router.post('/log', async (req, res) => {
    try {
        const newLog = new SleepLog(req.body);
        await newLog.save();
        res.json({ message: '✅ Sleep data logged' });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// Get Sleep Logs for a User
router.get('/:userId', async (req, res) => {
    try {
        const logs = await SleepLog.find({ userId: req.params.userId });
        res.json(logs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
