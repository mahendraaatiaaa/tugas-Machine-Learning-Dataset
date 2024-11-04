import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset
sms_data = pd.read_csv("spam.csv", encoding='latin-1')
sms_data = sms_data[['v1', 'v2']]
sms_data.columns = ['label', 'text']

# Preprocessing Data
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})  # Mengonversi label menjadi numerik

# Divided two parts as features and labels
X_features = sms_data['text']
y_labels = sms_data['label']

# divide the data test and data train
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = CountVectorizer(binary=True) 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Bernoulli Naive Bayes
model = BernoulliNB()
model.fit(X_train_vec, y_train)

#  Make the prediction
y_pred = model.predict(X_test_vec)

# Evaluation the Model - Accuracy
model_accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy of Bernoulli Naive Bayes from SPAMS dataset is:", model_accuracy)

# Evaluation the Model - Classification Report
model_classification_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print("The Classification Report of Bernoulli Naive Bayes from SPAMS dataset is:\n", model_classification_report)

# Evaluation the Model - F1 Score
model_f1_score = f1_score(y_test, y_pred, pos_label=1)
print("The F1 Score of Bernoulli Naive Bayes from SPAMS dataset is:", model_f1_score)

# Evaluation the Model - Confusion Matrix
model_confusion_matrix = confusion_matrix(y_test, y_pred)
print("The Confusion Matrix of Bernoulli Naive Bayes from SPAMS dataset is:\n", model_confusion_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(model_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Bernoulli Naive Bayes')
plt.show()


