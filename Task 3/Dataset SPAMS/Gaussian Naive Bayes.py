import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset
sms_data = pd.read_csv('spam.csv', encoding='latin-1')
sms_data = sms_data[['v1', 'v2']] 
sms_data.columns = ['label', 'message']

# Preprocessing Data
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})

# Count Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sms_data['message']).toarray()
y = sms_data['label'].values

# Divide the data test and data train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Make the prediction
y_pred = model.predict(X_test)

# Evaluation the Model - Accuracy
model_accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy of Gaussian Naive Bayes from SPAMS dataset is:", model_accuracy)

# Evaluation the Model - Classification Report
model_classification_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print("The Classification Report of Gaussian Naive Bayes from SPAMS dataset is:\n", model_classification_report)

# Evaluation the Model - F1 Score
model_f1_score = f1_score(y_test, y_pred, average='binary')
print("The F1 Score of Gaussian Naive Bayes from SPAMS dataset is:", model_f1_score)

# Evaluation the Model - Confusion Matrix
model_confusion_matrix = confusion_matrix(y_test, y_pred)
print("The Confusion Matrix of Gaussian Naive Bayes from SPAMS dataset is:\n", model_confusion_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(model_confusion_matrix, annot=True, cmap='Blues', fmt='g', 
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.show()
