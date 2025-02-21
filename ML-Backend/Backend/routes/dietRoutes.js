const express = require('express');
const DietPlan = require('../models/DietPlan');
const router = express.Router();

router.get('/', async (req, res) => {
    try {
        const plans = await DietPlan.find();
        if (plans.length === 0) return res.status(404).json({ message: "No diet plans found." });
        res.json(plans);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
