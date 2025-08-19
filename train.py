# train.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the Iris dataset
data = load_iris()
X = data.data       # Features
y = data.target     # Labels

# 2. Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Initialize the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate model on test set
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 6. Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
