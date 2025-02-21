const mongoose = require("mongoose");

const SleepLogSchema = new mongoose.Schema({
    hoursSlept: { type: Number, required: true },
    date: { type: Date, default: Date.now }
});

module.exports = mongoose.model("SleepLog", SleepLogSchema);
