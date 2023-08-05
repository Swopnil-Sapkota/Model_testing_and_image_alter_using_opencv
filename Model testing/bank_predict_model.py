import pandas as pd
import joblib

# Load correct_columns from file
correct_columns = joblib.load('correct_columns.pkl')

# Load the trained model
loaded_model = joblib.load('loan_classification_model.pkl')

# Create new data for testing
new_data = pd.DataFrame({
    'Age': [30],
    'Income': [75],
    'Education': [2],
    'Family': [3],
    'CCAvg': [1.5],
    'Mortgage': [0],
    'Securities Account': [0],
    'CD Account': [0],
    'Online': [1],
    'CreditCard': [0],
    'Gender': ['F'], 
    'Home Ownership': ['Home Mortgage'],  
})

# Encode the new data
new_data_encoded = pd.get_dummies(new_data, columns=['Gender', 'Home Ownership'], drop_first=True)

# Recreate the new data DataFrame with correct column names and order
new_data_encoded = new_data_encoded.reindex(columns=correct_columns, fill_value=0)

# Make predictions
predicted_classes = loaded_model.predict(new_data_encoded)

# Interpret the prediction
prediction_label = 'Accepted' if predicted_classes[0] == 1 else 'Rejected'

# Print prediction
print(f"The model predicts that the loan application will be: {prediction_label}")
