import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

DATASET = "copd_training_template.csv"
TARGET = "copd_risk_label"

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET)

print("\nDataset Loaded Successfully\n")
print("Shape :", df.shape)
print("\nColumns\n")
print(df.columns.tolist())

# =====================================================
# REMOVE DUPLICATES
# =====================================================

df = df.drop_duplicates()

# =====================================================
# REMOVE MISSING VALUES
# =====================================================

df = df.dropna()

# =====================================================
# CHECK TARGET
# =====================================================

if TARGET not in df.columns:
    raise Exception(f"\nTarget column '{TARGET}' not found!")

print("\nTarget Distribution\n")
print(df[TARGET].value_counts())

# =====================================================
# FEATURES
# =====================================================

X = df.drop(columns=[TARGET])

y = df[TARGET]

print("\nNumber of Features :", X.shape[1])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

print("\nTraining Random Forest...\n")

model = RandomForestClassifier(

    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42

)

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================================
# PREDICTIONS
# =====================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy : {:.2f}%".format(accuracy * 100))

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({

    "Feature": X.columns,
    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features\n")

print(importance.head(15))

importance.to_csv(
    "feature_importance.csv",
    index=False
)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully!")

print(os.path.abspath("model.pkl"))

# =====================================================
# SAMPLE PREDICTION
# =====================================================

print("\nTesting Saved Model...")

loaded_model = joblib.load("model.pkl")

sample = X.iloc[[0]]

prediction = loaded_model.predict(sample)[0]

probability = loaded_model.predict_proba(sample)[0]

print("\nPrediction :", prediction)

print("Confidence :", round(max(probability) * 100, 2), "%")

print("\nTraining Finished Successfully.")