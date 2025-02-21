const mongoose = require("mongoose");

const WaterLogSchema = new mongoose.Schema({
    liters: { type: Number, required: true },
    date: { type: Date, default: Date.now }
});

module.exports = mongoose.model("WaterLog", WaterLogSchema);
