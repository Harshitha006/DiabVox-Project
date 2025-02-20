const mongoose = require("mongoose");

const WaterLogSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  liters: { type: Number, required: true },
  date: { type: Date, default: Date.now },
});

module.exports = mongoose.model("WaterLog", WaterLogSchema);
