import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("data/jobs.csv")

# Features and labels
X = df["description"]
y = df["skill_category"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# Save artifacts
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model trained successfully!")