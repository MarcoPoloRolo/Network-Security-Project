from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Fetch Dataset
# Fetch the Spambase dataset from the UCI Machine Learning Repository
spambase = fetch_ucirepo(id=94)

# Separate the dataset into features (X) and labels/targets (y)
X = spambase.data.features  # Feature data (numerical values representing email characteristics)
y = spambase.data.targets   # Target data (1 = spam, 0 = non-spam)

# Step 2: Split Data into Training and Testing Sets
# Split the dataset into training (80%) and testing (20%) sets
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the sizes of the training and testing datasets
print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

# Step 3: Scale the Features
# Standardize the feature data to have zero mean and unit variance
# This helps the Logistic Regression model converge more efficiently
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform it
X_test_scaled = scaler.transform(X_test)        # Transform the test data using the same scaler

# Step 4: Train a Logistic Regression Model
# Initialize and train the Logistic Regression model
# max_iter is increased to ensure the model converges during training
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train.values.ravel())  # Train the model on the scaled training data

# Step 5: Evaluate the Model
# Predict the labels for the test data
y_pred = model.predict(X_test_scaled)

# Calculate and display the model's accuracy as a percentage
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print("\n========================")
print(f"Model Accuracy: {accuracy}%")
print("========================\n")

# Generate and display the classification report
# This includes precision, recall, F1-score, and support for each class
print("Classification Report:")
print("========================")
print(classification_report(y_test, y_pred))

# Step 6: Visualize Results Using a Confusion Matrix
# Create a confusion matrix to evaluate the performance of the model
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.xlabel('Predicted Labels')  # X-axis shows predicted classes
plt.ylabel('Actual Labels')    # Y-axis shows actual classes
plt.title('Confusion Matrix')  # Title of the heatmap
plt.show()

# Step 7: Test with New Data
# Simulated test emails with 57 features each
# First email represents a likely spam email, second email represents non-spam
test_emails = [
    [0, 0.64, 0.64, 0, 0, 0, 0.32, 0.64, 0, 0.32, 0.64, 0, 0, 0.64, 0, 0.32, 0.32, 0, 0, 0, 3.75, 61, 278] + [0]*34,  # Simulated spam
    [0]*57  # Simulated non-spam with all features as 0
]

# Convert test_emails into a DataFrame with feature names matching the original dataset
test_emails_df = pd.DataFrame(test_emails, columns=X.columns)

# Scale the test data using the same scaler used for the training data
test_emails_scaled = scaler.transform(test_emails_df)

# Predict the labels for the new test emails
test_predictions = model.predict(test_emails_scaled)

# Display the predictions for the new test emails
print("\nTest Email Predictions:")
for i, pred in enumerate(test_predictions):
    print(f"Test Email {i + 1}: {'Spam' if pred == 1 else 'Non-Spam'}")

# Final separator to indicate the end of the program's output
print("\n========================")
print("End of Model Evaluation")
print("========================")
