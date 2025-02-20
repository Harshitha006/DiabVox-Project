const express = require('express');
const DietPlan = require('../models/DietPlan');
const router = express.Router();

// Generate Diet Plan
router.post('/generate', async (req, res) => {
    try {
        const { userId, glucoseLevel } = req.body;
        const recommendedFoods = glucoseLevel > 150 ? ['Vegetables', 'Whole Grains', 'Lean Protein'] : ['Fruits', 'Oats', 'Nuts'];
        const restrictedFoods = glucoseLevel > 150 ? ['Sugary Drinks', 'White Bread', 'Processed Foods'] : ['Excessive Sugar', 'Fried Foods'];

        const newPlan = new DietPlan({ userId, recommendedFoods, restrictedFoods });
        await newPlan.save();
        res.json({ message: '✅ Diet plan generated', newPlan });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// Get Diet Plan by User ID
router.get('/:userId', async (req, res) => {
    try {
        const plan = await DietPlan.findOne({ userId: req.params.userId });
        if (!plan) return res.status(404).json({ message: '❌ No diet plan found' });
        res.json(plan);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
