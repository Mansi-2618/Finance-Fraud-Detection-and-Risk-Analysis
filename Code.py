import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("/content/Trans_Dataset.csv")
df["isFraud"].fillna(0, inplace=True)
# Feature Selection
features = ["Amount", "Old Balance(Sender)", "New Balance(Sender)", "Old Balance(Reciever)",
"New Balance(Reciever)"]
X = df[features]
y = df["isFraud"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict Fraud
df["fraud_prediction"] = model.predict(X)
df.to_csv("fraud_predictions.csv", index=False)

print("Fraud predictions saved successfully to fraud_predictions.csv")