import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load CSV files
disease_df = pd.read_csv('Disease.csv')
drug_df = pd.read_csv('Drug.csv')

# Preprocess data
X = disease_df.drop('prognosis', axis=1)
y = disease_df['prognosis']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model
joblib.dump(clf, 'models/model.pkl')

print("Model trained and saved successfully.")
