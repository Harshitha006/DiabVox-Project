import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Diabetes-dataset.csv")
df.fillna(df.mean(), inplace=True)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

pickle.dump(model, open("lr.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("🎉 Model retrained and saved as lr.pkl & scaler.pkl!")
