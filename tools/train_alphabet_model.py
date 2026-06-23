import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")
data = pd.read_csv("landmark_dataset.csv")

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gesture_model.pkl")
print("Model saved to models/gesture_model.pkl")