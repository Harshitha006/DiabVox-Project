import pandas as pd

# Load the dataset
df = pd.read_csv("Diabetes-dataset.csv")

# Print all columns used for training
print("Columns in dataset:", df.columns.tolist())

# Check the exact columns used for training
X = df.drop(columns=["Outcome"])  # Remove target column
print("Columns used for training:", X.columns.tolist())
