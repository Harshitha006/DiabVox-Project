const mongoose = require("mongoose");
const User = require("./models/User");
const bcrypt = require("bcryptjs");
require('dotenv').config();

// MongoDB Connection
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => console.log("✅ MongoDB Connected"))
  .catch(err => console.log(err));

const seedUsers = async () => {
    try {
        await User.deleteMany();
        
        const users = [
            { name: "Alice", email: "alice@example.com", password: await bcrypt.hash("password123", 10) },
            { name: "Bob", email: "bob@example.com", password: await bcrypt.hash("securepass", 10) },
        ];

        await User.insertMany(users);
        console.log("✅ Users added successfully!");
        mongoose.connection.close();
    } catch (error) {
        console.error("❌ Error inserting users:", error);
        mongoose.connection.close();
    }
};

// Run
seedUsers();
