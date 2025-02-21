const mongoose = require("mongoose");

const DietPlanSchema = new mongoose.Schema({
    recommendedFoods: [String],
    restrictedFoods: [String]
});

module.exports = mongoose.model("DietPlan", DietPlanSchema);
