# satisfaction_prediction.py

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
df = pd.read_csv("test.csv")

# 2. Clean column names
df.columns = df.columns.str.strip()

# 3. Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 4. Convert data types if needed
df['Age'] = df['Age'].astype(int)
df['Flight Distance'] = df['Flight Distance'].astype(int)

# 5. Identify features
cat_features = df.select_dtypes(include=['object']).columns
num_features = df.select_dtypes(include=['int64', 'float64']).columns

# 6. Encode categorical features
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 7. Remove outliers using IQR
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# 8. Features and Target
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# 11. Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

# 12. Save best model (Random Forest)
best_model = models["Random Forest"]
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# 13. Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
