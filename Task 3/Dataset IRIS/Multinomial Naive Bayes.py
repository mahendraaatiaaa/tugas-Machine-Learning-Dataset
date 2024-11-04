from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
iris_data = pd.read_csv("Iris.csv")

# Split up the dataset as features and target
X_features = iris_data.drop(['Id', 'Species'], axis=1)
Y_target = iris_data['Species']

# Convert target labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y_target)

# Scaling the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_features)

# Split up the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Make the predictions
y_pred = model.predict(X_test)

# Evaluate the model - Accuracy
model_accuracy = accuracy_score(y_test, y_pred)
print(f"The Accuracy of Multinomial Naive Bayes from IRIS dataset is: {model_accuracy:.2f}")

# Evaluate the model - Classification Report
print("The Classification Report of Multinomial Naive Bayes from IRIS dataset is:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# Evaluate the model - F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"The F1 Score of Multinomial Naive Bayes from IRIS dataset is: {f1:.2f}")

# Evaluasi - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("The Confusion Matrix of Multinomial Naive Bayes from IRIS dataset is:")
print(cm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Multinomial Naive Bayes')
plt.show()
