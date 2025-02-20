// routes/voiceRoutes.js
const express = require('express');
const router = express.Router();

router.post('/process', (req, res) => {
    const { voiceCommand } = req.body;
    if (voiceCommand.toLowerCase().includes('track health')) {
        res.json({ response: 'Fetching your latest health stats.' });
    } else {
        res.json({ response: 'Command not recognized.' });
    }
});

module.exports = router;

