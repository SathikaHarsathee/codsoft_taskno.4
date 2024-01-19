import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sms_data = pd.read_csv('spam.csv', encoding='latin-1')
sms_data = sms_data[['v1', 'v2']] 
sms_data.columns = ['label', 'message']


sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})
sms_data.dropna(subset=['label'], inplace=True)


plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=sms_data, palette='spring_r', hue='label', legend=False)
plt.title('Distribution of Labels', fontsize=14)
plt.show()


X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(
    sms_data['message'], sms_data['label'], test_size=0.2, random_state=42
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_data_train)
X_test_tfidf = tfidf_vectorizer.transform(X_data_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_data_train)


y_pred = naive_bayes.predict(X_test_tfidf)

cm = confusion_matrix(y_data_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()


accuracy = accuracy_score(y_data_test, y_pred)
classification_rep = classification_report(y_data_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


sms_data['message_length'] = sms_data['message'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data=sms_data, x='message_length', hue='label', kde=True, multiple='stack', palette='viridis')
plt.title('Distribution of Message Lengths by Label', fontsize=14)
plt.show()

from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(naive_bayes, X_train_tfidf, y_data_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cross_val_scores)


alphas = [0.1, 1.0, 10.0]
for alpha in alphas:
    nb_model = MultinomialNB(alpha=alpha)
    nb_model.fit(X_train_tfidf, y_data_train)
    y_pred_cv = nb_model.predict(X_test_tfidf)
    accuracy_cv = accuracy_score(y_data_test, y_pred_cv)
    print(f"Alpha={alpha}, Accuracy: {accuracy_cv}")

