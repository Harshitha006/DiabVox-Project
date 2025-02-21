const express = require("express");
const router = express.Router();
const mongoose = require("mongoose");

// ✅ Define Mongoose Schema for Dataset
const diabetesSchema = new mongoose.Schema({}, { strict: false });
const DiabetesRecord = mongoose.model("DiabetesRecord", diabetesSchema, "diabetes_records");

// ✅ API to Fetch Diabetes Records
router.get("/", async (req, res) => {
    try {
        const records = await DiabetesRecord.find();
        res.json(records);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
