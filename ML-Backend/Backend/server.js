const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();

// Middleware
app.use(express.json());
app.use(cors());

// ✅ MongoDB Connection
mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log('✅ MongoDB Connected'))
.catch(err => console.error('❌ MongoDB Connection Error:', err));

// ✅ Import Routes
const userRoutes = require('./routes/userRoutes');
const healthRoutes = require('./routes/healthRoutes');
const dietRoutes = require('./routes/dietRoutes');
const sleepRoutes = require('./routes/sleepRoutes');
const waterRoutes = require('./routes/waterRoutes');
const aiRoutes = require('./routes/aiRoutes');          // ML API Calls
const diabetesRoutes = require('./routes/diabetesRoutes'); // Fetch stored dataset

// ✅ Use Routes
app.use('/api/users', userRoutes);
app.use('/api/health', healthRoutes);
app.use('/api/diet', dietRoutes);
app.use('/api/sleep', sleepRoutes);
app.use('/api/water', waterRoutes);
app.use('/api/ai', aiRoutes);
app.use('/api/diabetes', diabetesRoutes);  // New Route to Get Dataset

app.get('/', (req, res) => {
    res.send('🚀 DiabVox API is Running!');
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`✅ Server running on port ${PORT}`);
});
