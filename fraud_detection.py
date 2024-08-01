# libraries used 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Style configuration for views
sns.set(style="whitegrid")

# load data
data = pd.read_csv('transaction_dataset.csv')

# Check the distribution of classes
sns.countplot(x='Class', data=data)
plt.title('Distribuição das Classes')
plt.show()

# Separate data into features and targets
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Training a Random Forest model
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_scaled, y_train)

# Forecasting with Logistic Regression
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Forecasting with Random Forest
y_pred_rf = rf_clf.predict(X_test_scaled)

# Classification reports
print("Classification Report - Logistic Regression")
print(classification_report(y_test, y_pred_log_reg))

print("Classification Report - Random Forest")
print(classification_report(y_test, y_pred_rf))

# Confusion matrices
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix - Logistic Regression')
plt.show()

sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix - Random Forest')
plt.show()

# ROC and AUC curves
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test_scaled)[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_clf.predict_proba(X_test_scaled)[:,1])

plt.figure()
plt.plot(fpr_log_reg, tpr_log_reg, color='blue', lw=2, label='Logistic Regression (AUC = %0.2f)' % auc(fpr_log_reg, tpr_log_reg))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % auc(fpr_rf, tpr_rf))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
