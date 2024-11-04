import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris_data = pd.read_csv("Iris.csv")

# Split up the dataset as features and target
X_features = iris_data.drop(['Species', 'Id'], axis=1)  # Menghapus kolom 'Species' dan 'Id' dari fitur
y_target = iris_data['Species']  # Kolom target adalah 'Species'

# Convert target labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_target)

# Split up the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_features, y_encoded, test_size=0.2, random_state=42)

# Gaussian Naive Bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)

# Make the predictions
predictions = gaussian_nb.predict(X_test)

# Evaluate the model - Accuracy
model_accuracy = accuracy_score(y_test, predictions)
print(f"The Accuracy of Gaussian Naive Bayes from IRIS dataset is: {model_accuracy:.2f}")

# Evaluate the model - Classification Report
class_report = classification_report(y_test, predictions, target_names=label_encoder.classes_)
print("The Classification Report of Gaussian Naive Bayes from IRIS dataset is:")
print(class_report)

# Menghitung confusion matrix
cm = confusion_matrix(y_test, predictions)
print("The Confusion Matrix of Gaussian Naive Bayes from IRIS dataset is:")
print(cm)

# Menghitung F1 score
f1 = f1_score(y_test, predictions, average='weighted')
print(f"The F1 Score of Gaussian Naive Bayes from IRIS dataset is: {f1:.2f}")

# Menampilkan confusion matrix dengan visualisasi
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.show()
