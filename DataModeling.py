import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, ConfusionMatrixDisplay, confusion_matrix, \
    roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load data
data = pd.read_csv('Bicycle_Thefts.csv')

# Drop unnecessary columns
data = data.drop(columns=['OBJECTID', 'EVENT_UNIQUE_ID'])

# Handle missing data
missing_counts = data.isnull().sum()
missing_percentage = data.isnull().mean() * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_percentage
})

# Fill missing data using media, mode or unknown for low percentage of missing data or no correlate data
bike_speed_median = data['BIKE_SPEED'].median()
bike_cost_median = data['BIKE_COST'].median()
unknown_placeholder = "unknown"

data.fillna({"BIKE_SPEED": bike_speed_median}, inplace=True)
data.fillna({"BIKE_COST": bike_cost_median}, inplace=True)
data.fillna({"BIKE_MAKE": unknown_placeholder}, inplace=True)
data.fillna({"BIKE_COLOUR": unknown_placeholder}, inplace=True)
data.fillna({"BIKE_MODEL": unknown_placeholder}, inplace=True)

# Encoding categorical values
categorical_columns = data.select_dtypes(include=['object']).columns.drop('STATUS')
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Encode target variable
label_encoder = LabelEncoder()
data['STATUS'] = label_encoder.fit_transform(data['STATUS'])

# Normalize numerical features
scaler = StandardScaler()
numerical_columns = ['OCC_HOUR', 'REPORT_HOUR', 'BIKE_COST',
                     'LONG_WGS84', 'LAT_WGS84', 'x', 'y', 'BIKE_SPEED']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Split features and target
X = data.drop(columns=['STATUS'])
y = data['STATUS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Logistic regression model
logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logistic_model.fit(X_train_balanced_scaled, y_train_balanced)

y_pred = logistic_model.predict(X_test_scaled)
y_pred_prob = logistic_model.predict_proba(X_test_scaled)

# Scores
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Logistic Regression")
# ROC Curve
for i, class_label in enumerate(label_encoder.classes_):
    fpr, tpr, thresholds = roc_curve(y_test == i, y_pred_prob[:, i])
    auc_score = roc_auc_score(y_test == i, y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"{class_label} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Logistic Regression')
plt.legend()
plt.show()

# Decesion tree model
tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
tree_model.fit(X_train_balanced, y_train_balanced)

y_pred_tree = tree_model.predict(X_test)
y_pred_prob_tree = tree_model.predict_proba(X_test)

# Scores
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))
# Confusion Matrix
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
disp_tree = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_tree, display_labels=label_encoder.classes_)
disp_tree.plot(cmap='Blues')
plt.title("Confusion Matrix: Decision Tree")
# ROC Curve
for i, class_label in enumerate(label_encoder.classes_):
    fpr, tpr, thresholds = roc_curve(y_test == i, y_pred_prob_tree[:, i])
    auc_score = roc_auc_score(y_test == i, y_pred_prob_tree[:, i])
    plt.plot(fpr, tpr, label=f"{class_label} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Decision Tree')
plt.legend()
plt.show()

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', max_features='sqrt',
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)

y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)

# Scores
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=label_encoder.classes_)
disp_rf.plot(cmap='Blues')
plt.title("Confusion Matrix: Random Forest")
# ROC Curve
for i, class_label in enumerate(label_encoder.classes_):
    fpr, tpr, thresholds = roc_curve(y_test == i, y_pred_prob_rf[:, i])
    auc_score = roc_auc_score(y_test == i, y_pred_prob_rf[:, i])
    plt.plot(fpr, tpr, label=f"{class_label} (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Random Forest')
plt.legend()
plt.show()


model_columns = X.columns
with open('model_columns.pkl', 'wb') as file:
    pickle.dump(model_columns, file)
print("Model columns saved as 'model_columns.pkl'")

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
print("Model saved as 'random_forest_model.pkl'")