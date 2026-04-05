import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ----------------------------
# Load Dataset
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "electronics_products_pricing.csv")

df = pd.read_csv(file_path)

print("Original Shape:", df.shape)

# ----------------------------
# Keep Important Columns
# ----------------------------
df = df[[
    "brand",
    "categories",
    "primaryCategories",
    "prices.availability",
    "prices.condition",
    "prices.isSale",   
    "weight",
    "price"
]]

# ----------------------------
# Clean Price
# ----------------------------
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# ----------------------------
# Clean Weight
# ----------------------------
df["weight"] = df["weight"].astype(str)
df["weight"] = df["weight"].str.extract("(\d+\.?\d*)")
df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

# ----------------------------
# Handle Missing Values
# ----------------------------
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna("Unknown", inplace=True)

# ----------------------------
# Encode Categorical Columns
# ----------------------------
encoders = {}

for column in [
    "brand",
    "categories",
    "primaryCategories",
    "prices.availability",
    "prices.condition",
    "prices.isSale"
]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    encoders[column] = le

joblib.dump(encoders, os.path.join(BASE_DIR, "encoder.pkl"))

# ----------------------------
# Define Features & Target
# ----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Model
# ----------------------------
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ----------------------------
# Save Model
# ----------------------------
joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))

print("Model and encoders saved successfully!")