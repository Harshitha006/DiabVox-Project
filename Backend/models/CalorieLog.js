const mongoose = require("mongoose");

const CalorieLogSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  calories: { type: Number, required: true },
  food: { type: String, required: true },
  date: { type: Date, default: Date.now },
});

module.exports = mongoose.model("CalorieLog", CalorieLogSchema);
