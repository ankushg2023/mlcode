import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv("D:/Programming/Python/Admission_Predict.csv")

# Convert 'Admitted' to binary (1 or 0)
threshold = 0.5  # Adjust the threshold based on your preference
df['Admitted'] = df['Admitted'].apply(lambda x: 1 if x >= threshold else 0)

# Select relevant features (GRE score and Undergraduate GPA) and the target variable (Admitted)
X = df[['GRE Score', 'CGPA']]
y = df['Admitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['GRE Score', 'Undergraduate GPA'], class_names=['Not Admitted', 'Admitted'], filled=True, rounded=True)
plt.show()

print(df.to_string())
