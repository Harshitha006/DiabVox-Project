import pandas as pd
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["diabvox"]

df = pd.read_csv("Diabetes-dataset.csv")

data = df.to_dict(orient="records")
db.diabetes_records.insert_many(data)

print("🎉 Diabetes dataset inserted into MongoDB!")
