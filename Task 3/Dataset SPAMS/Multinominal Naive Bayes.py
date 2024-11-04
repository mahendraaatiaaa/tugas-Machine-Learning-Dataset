import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load the SMS Spam Collection Dataset
sms_data = pd.read_csv("spam.csv", encoding='latin-1')  # url: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

# Preprocess the data
sms_data = sms_data[['v1', 'v2']]
sms_data = sms_data.rename(columns={'v1': 'label', 'v2': 'text'})

# Split the data into features and labels
X_features = sms_data['text']
y_labels= sms_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

# EDA 1: Distribution of Classes
class_distribution = sms_data['label'].value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'])
plt.title('Distribution of Spam and Ham Messages')
plt.show()

# Generate WordCloud for Spam Messages
spam_text = ' '.join(sms_data[sms_data['label'] == 'spam']['text'])
spam_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white', random_state=42).generate(spam_text)

# Generate WordCloud for Ham Messages
ham_text = ' '.join(sms_data[sms_data['label'] == 'ham']['text'])
ham_wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white', random_state=42).generate(ham_text)

# Plot the WordClouds
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Spam Messages')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Ham Messages')
plt.axis('off')

plt.tight_layout()
plt.show()

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
mnb = MultinomialNB(alpha=0.8, fit_prior=True)
mnb.fit(X_train_vec, y_train)

# Evaluate the model - Accuracy
y_pred = mnb.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("The Accuracy of Multinominal Naive Bayes from SPAMS dataset is:", accuracy)

# Evaluate the model - Classification Report
print("The Classification Report of Multinominal Naive Bayes from SPAMS dataset is:\n", classification_report(y_test, y_pred))

# Evaluate the model - F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print("The F1 Score of Multinominal Naive Bayes from SPAMS dataset is:", f1)

# Evaluate the model - Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print("The Confusion Matrix of Multinominal Naive Bayes from SPAMS dataset is:\n", confusion)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Multinominal Naive Bayes')
plt.show()