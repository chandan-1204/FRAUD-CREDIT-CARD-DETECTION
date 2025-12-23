import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE


# ===============================
# 1. LOAD DATASET
# ===============================
data = pd.read_csv("data/creditcard.csv")
print("Dataset Loaded Successfully")

# ===============================
# 2. EXPLORATORY DATA ANALYSIS
# ===============================
print("\nClass Distribution:")
print(data['Class'].value_counts())

# ===============================
# 3. FEATURE SCALING
# ===============================
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# ===============================
# 4. SPLIT FEATURES & TARGET
# ===============================
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 5. HANDLE IMBALANCED DATA (SMOTE)
# ===============================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ===============================
# 6. MODEL 1: ISOLATION FOREST
# ===============================
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.0017,
    random_state=42
)

iso_forest.fit(X_train)

y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

print("\nIsolation Forest Results:")
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))

# ===============================
# 7. MODEL 2: LOGISTIC REGRESSION (WITH SMOTE)
# ===============================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_smote, y_train_smote)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

roc_auc = roc_auc_score(y_test, y_prob_lr)
print("ROC-AUC Score:", roc_auc)

# ===============================
# 8. SAVE MODEL
# ===============================
joblib.dump(lr_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and Scaler Saved Successfully!")
