const mongoose = require("mongoose");

const HealthLogSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  bloodGlucose: { type: Number, required: true },
  insulinLevel: { type: Number, required: true },
  date: { type: Date, default: Date.now },
});

module.exports = mongoose.model("HealthLog", HealthLogSchema);
