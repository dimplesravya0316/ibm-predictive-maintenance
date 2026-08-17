import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
df = pd.read_csv('ai4i2020.csv')

# Clean column names (strip whitespace / brackets)
df.columns = df.columns.str.strip().str.replace('[', '', regex=False).str.replace(']', '', regex=False)

# 2. Feature Engineering
df['Temp_Difference'] = df['Process temperature K'] - df['Air temperature K']
df['Power_W'] = df['Torque Nm'] * (df['Rotational speed rpm'] * (2 * np.pi / 60))
df['Overstrain_Index'] = df['Tool wear min'] * df['Torque Nm']

# Select features and targets
feature_cols = [
    'Type', 'Air temperature K', 'Process temperature K', 
    'Rotational speed rpm', 'Torque Nm', 'Tool wear min',
    'Temp_Difference', 'Power_W', 'Overstrain_Index'
]
target_col = 'Machine failure'

X = df[feature_cols].copy()
y = df[target_col]

# Encode categorical column 'Type' (L, M, H)
type_encoder = LabelEncoder()
X['Type'] = type_encoder.fit_transform(X['Type'])

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# 6. Model Training
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# 7. Model Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# 8. Save Artifacts
artifacts = {
    'model': model,
    'scaler': scaler,
    'type_encoder': type_encoder,
    'feature_cols': feature_cols
}
joblib.dump(artifacts, 'model.pkl')
print("\nModel saved successfully as 'model.pkl'!")