import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Step 1: Data Preprocessing

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the Excel file
excel_file_path = os.path.join(script_dir, 'Bank_loan_data.xlsx')

# Read the Excel file
data = pd.read_excel(excel_file_path)
data.dropna(inplace=True)
data = pd.get_dummies(data, columns=['Gender', 'Home Ownership'])
scaler = StandardScaler()
numerical_features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage', 'ZIP Code']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Remove rows with empty target values
data = data[data['Personal Loan'] != ' ']

# Convert target variable to int
data['Personal Loan'] = data['Personal Loan'].astype(int)

# Step 4: Model Selection and Training
X = data.drop(['ID', 'Personal Loan'], axis=1)
y = data['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 6: Save the Model
joblib.dump(model, 'loan_classification_model.pkl')

# Save correct_columns to a file
correct_columns = X_train.columns.tolist()
joblib.dump(correct_columns, 'correct_columns.pkl')
