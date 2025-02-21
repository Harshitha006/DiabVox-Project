const express = require('express');
const WaterLog = require('../models/WaterLog');
const router = express.Router();

router.get('/', async (req, res) => {
    try {
        const logs = await WaterLog.find();
        if (logs.length === 0) return res.status(404).json({ message: "No water logs found." });
        res.json(logs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
