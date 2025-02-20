const mongoose = require("mongoose");

const DietPlanSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  recommendedFoods: [String],
  restrictedFoods: [String],
});

module.exports = mongoose.model("DietPlan", DietPlanSchema);
