import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris_data = pd.read_csv('Iris.csv')

# Split up the dataset as features and target
X_features = iris_data.drop(columns=['Id', 'Species'])  # Menghapus kolom 'Id' dan 'Species' dari fitur
y_target = iris_data['Species']  # Menyimpan kolom target 'Species'

# Convert target labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_target)

# Binarize the features
binarizer = Binarizer(threshold=X_features.mean().mean())  # Menggunakan rata-rata dari semua fitur sebagai threshold
X_binarized = binarizer.fit_transform(X_features)

# Split up the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_binarized, y_encoded, test_size=0.2, random_state=42
)

# Bernoulli Naive Bayes
bernoulli_nb_model = BernoulliNB()
bernoulli_nb_model.fit(X_train, y_train)

# Make the predictions
predictions = bernoulli_nb_model.predict(X_test)

# Evaluate the model - Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"The Accuracy with Bernoulli Naive Bayes from IRIS dataset is: {accuracy:.2f}")

# Evaluate the model - Classification Report
print("The Classification Report with Bernoulli Naive Bayes from IRIS dataset is:")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

# Menghitung F1 score
f1 = f1_score(y_test, predictions, average='weighted')
print(f"The F1 Score with Bernoulli Naive Bayes from IRIS dataset is: {f1:.2f}")

# Menghitung confusion matrix
cm = confusion_matrix(y_test, predictions)
print("The Confusion Matrix with Bernoulli Naive Bayes from IRIS dataset is:")
print(cm)

#Visualize of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Bernoulli Naive Bayes')
plt.show()

