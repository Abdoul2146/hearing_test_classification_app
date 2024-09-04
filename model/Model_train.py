import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import joblib

# Load your dataset
data = pd.read_csv("hearing_test.csv")

# 'X' contains my features and 'y' contains my target variable
X = data.drop('test_result', axis=1)  # Features
y = data['test_result']  # Target variable

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(model, 'hearing_test_model.pkl')

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("Accuracy:", accuracy)
print("Precision: ", precision)
print("recall: ", recall)
print("f1 score: ", f1)
print('confusion matrix: ', conf_matrix)
