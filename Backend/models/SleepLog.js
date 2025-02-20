const mongoose = require("mongoose");

const SleepLogSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  hoursSlept: { type: Number, required: true },
  date: { type: Date, default: Date.now },
});

module.exports = mongoose.model("SleepLog", SleepLogSchema);
