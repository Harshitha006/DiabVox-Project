// const express = require('express');
// const mongoose = require('mongoose');
// const cors = require('cors');
// require('dotenv').config();

// const app = express();

// // Middleware
// app.use(express.json());
// app.use(cors());

// // MongoDB Connection
// mongoose.connect(process.env.MONGO_URI, {
//     useNewUrlParser: true,
//     useUnifiedTopology: true,
// })
// .then(() => console.log('✅ MongoDB Connected'))
// .catch(err => console.error('❌ MongoDB Connection Error:', err));

// // Routes
// app.use('/api/users', require('./routes/userRoutes'));
// app.use('/api/health', require('./routes/healthRoutes'));
// app.use('/api/diet', require('./routes/dietRoutes'));
// app.use('/api/ai', require('./routes/aiRoutes'));
// app.use('/api/voice', require('./routes/voiceRoutes'));

// // Test Route
// app.get('/', (req, res) => {
//     res.send('🚀 DiabVox Backend API is Running!');
// });

// // Start Server
// const PORT = process.env.PORT || 5000;
// app.listen(PORT, () => {
//     console.log(`✅ Server running on port ${PORT}`);
// });
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();

// Middleware
app.use(express.json());
app.use(cors());

// MongoDB Connection
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('✅ MongoDB Connected'))
.catch(err => console.error('❌ MongoDB Connection Error:', err));

// Importing Routes
const userRoutes = require('./routes/userRoutes');
const healthRoutes = require('./routes/healthRoutes');
const dietRoutes = require('./routes/dietRoutes');
const sleepRoutes = require('./routes/sleepRoutes');
const waterRoutes = require('./routes/waterRoutes');

// Using Routes
app.use('/api/users', userRoutes);
app.use('/api/health', healthRoutes);
app.use('/api/diet', dietRoutes);
app.use('/api/sleep', sleepRoutes);
app.use('/api/water', waterRoutes);

app.get('/', (req, res) => {
    res.send('🚀 DiabVox API is Running!');
});

// Test Route to check all routes
app.get('/api/test', (req, res) => {
    res.json({
        message: '🚀 API is working!',
        endpoints: [
            { route: '/api/users', description: 'User authentication and data' },
            { route: '/api/health', description: 'Health logging (glucose, insulin)' },
            { route: '/api/diet', description: 'Diet plan recommendations' },
            { route: '/api/sleep', description: 'Sleep tracking' },
            { route: '/api/water', description: 'Water intake tracking' },
        ]
    });
});

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`✅ Server running on port ${PORT}`);
});
