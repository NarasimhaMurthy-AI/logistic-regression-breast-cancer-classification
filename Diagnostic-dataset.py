# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display sample data
print("Dataset sample:")
print(df.head())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# Optional: Plot sigmoid function (for explanation)
x_vals = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-x_vals))

plt.plot(x_vals, sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("Input (x)")
plt.ylabel("Sigmoid(x)")
plt.grid()
plt.show()

# Optional: Threshold tuning (e.g., 0.6 instead of 0.5)
custom_threshold = 0.6
y_pred_thresh = (y_prob >= custom_threshold).astype(int)
cm_thresh = confusion_matrix(y_test, y_pred_thresh)
sns.heatmap(cm_thresh, annot=True, fmt='d', cmap='Oranges')
plt.title(f'Confusion Matrix (Threshold = {custom_threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save test results with predictions
X_test_cleaned = pd.DataFrame(X_test_scaled, columns=X.columns)
X_test_cleaned['Actual'] = y_test.values
X_test_cleaned['Predicted'] = y_pred
X_test_cleaned['Probability'] = y_prob
X_test_cleaned.to_csv(r"D:\AIML PROJECTS\diagnostic_results.csv", index=False)

