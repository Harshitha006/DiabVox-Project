const express = require('express');
const SleepLog = require('../models/SleepLog');
const router = express.Router();

router.get('/', async (req, res) => {
    try {
        const logs = await SleepLog.find();
        if (logs.length === 0) return res.status(404).json({ message: "No sleep logs found." });
        res.json(logs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
