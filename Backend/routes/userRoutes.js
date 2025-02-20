const express = require('express');
const bcrypt = require('bcryptjs');
const User = require('../models/User');
const router = express.Router();

// Register User
router.post('/register', async (req, res) => {
    try {
        const salt = await bcrypt.genSalt(10);
        req.body.password = await bcrypt.hash(req.body.password, salt);
        const newUser = new User(req.body);
        await newUser.save();
        res.json({ message: '✅ User registered successfully' });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// Get All Users
router.get('/', async (req, res) => {
    try {
        const users = await User.find();
        res.json(users);
    } catch (error) {
        res.status(500).json({ message: "❌ Server Error", error: error.message });
    }
});

module.exports = router;
