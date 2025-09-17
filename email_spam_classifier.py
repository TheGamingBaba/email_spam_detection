import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns




data= pd.read_csv('data/email/email.csv')
# print(data.head())
# print(data.columns)

# Remove leading/trailing spaces from column names and values
data.columns = data.columns.str.strip()
data['Category'] = data['Category'].astype(str).str.strip()

#keeping data that is only in the ham/ spam category
data = data[data['Category'].isin(['ham', 'spam'])]
data['Category'] = data['Category'].map({'ham':0,'spam':1}).astype(int)

#data preprocessing 
def text_cleaning(text):
    text = text.lower() #converts to all character to lower case
    text = re.sub(r'http\S+|www\S+|https\S+','',text) #removes all the url bits begining with the cases and and stopping when It sees a white space
    text = text.translate(str.maketrans('','',string.punctuation)) #removes all the punctuation from text
    text = re.sub(r'\d+', '', text) #removes numbers
    text = re.sub(r'\s+', ' ', text).strip() #removes extra spaces
    text = re.sub(r'\S+@\S+', '', text)  # removes email addresses
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)


    return text

data['Message'] = data['Message'].apply(text_cleaning)

# print(data.head(10))

#applying CountVectorizer 

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])  # X = numeric features
y = data['Category'] 

print("Vocabulary:", vectorizer.get_feature_names_out())

print("Vectorized form:\n", X.toarray())
#splitting the data set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#calling the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train,y_train) #model fitting 
 
y_pred = model.predict(X_test) #model prediction 

#testing efficacy of model across different parameters: 
# Accuracy → percentage of correct predictions
accuracy = accuracy_score(y_test, y_pred)

# Precision → of all messages classified as spam, how many were actually spam?
precision = precision_score(y_test, y_pred)

# Recall → of all actual spam messages, how many did we correctly detect?
recall = recall_score(y_test, y_pred)

# F1-score → balance between precision and recall
f1 = f1_score(y_test, y_pred)

# Confusion matrix → table showing true/false positives/negatives
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nConfusion Matrix:\n", cm)



#Visualizing the Confusion Matrix: 


# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




