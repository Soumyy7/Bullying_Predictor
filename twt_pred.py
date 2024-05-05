import pandas as pd
import numpy as np
import nltk
import string
import re
import pickle

nltk.download('stopwords')
nltk.download('wordnet')

# train_data = pd.read_csv('train.csv')
train_data = pd.read_csv('cyberbullying_tweets_4k.csv')
test_data = pd.read_csv('test.csv')

data = pd.concat([train_data, test_data], axis=0)

df = data.copy()

df['label'].value_counts()


# Removing URLs
def remove_url(text):
    return re.sub(r"http\S+", "", text)

#Removing Punctuations
def remove_punct(text):
    new_text = []
    for t in text:
        if t not in string.punctuation:
            new_text.append(t)
    return ''.join(new_text)


#Tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')



#Removing Stop words
from nltk.corpus import stopwords
def remove_sw(text):
    new_text = []
    for t in text:
        if t not in stopwords.words('english'):
            new_text.append(t)
    return new_text

#Lemmatizaion
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    new_text = []
    for t in text:
        lem_text = lemmatizer.lemmatize(t)
        new_text.append(lem_text)
    return new_text

df['tweet'] = df['tweet'].apply(lambda t: remove_url(t))

df['tweet'] = df['tweet'].apply(lambda t: remove_punct(t))

df['tweet'] = df['tweet'].apply(lambda t: tokenizer.tokenize(t.lower()))

df['tweet'] = df['tweet'].apply(lambda t: remove_sw(t))

df['tweet'] = df['tweet'].apply(lambda t: word_lemmatizer(t))

features_set = df.copy()

train_set = features_set.iloc[:len(train_data), :]

test_set = features_set.iloc[len(train_data):, :]

X = train_set['tweet']


for i in range(0, len(X)):
    X.iloc[i] = ' '.join(X.iloc[i])


Y = train_set['label']


from sklearn.feature_extraction.text import TfidfVectorizer

TfidV = TfidfVectorizer()

X = TfidV.fit_transform(X)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1234)


# from lightgbm import LGBMClassifier

# lgb = LGBMClassifier(scale_pos_weight=3, num_class = 5)

# lgb.fit(X, Y)

# #lgb.fit(x_train, y_train)

# y_predict_lgb = lgb.predict(x_test)

# from sklearn.metrics import confusion_matrix, f1_score

# cm_lgb = confusion_matrix(y_test, y_predict_lgb)

# # f1_lgb = f1_score(y_test, y_predict_lgb)

# f1_lgb = f1_score(y_test, y_predict_lgb, average='micro')

# score_lgb = lgb.score(x_test, y_test)   


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

# Create the individual models
nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=25, random_state=18)
lr = LogisticRegression(max_iter=500,C = 0.45)

# Create the ensemble model
ensemble = VotingClassifier(estimators=[('nb', nb), ('rf', rf), ('lr', lr)], voting='soft')

# Train the model
ensemble.fit(X, Y)

# Make predictions
y_predict_ensemble = ensemble.predict(x_test)

# Compute the confusion matrix
cm_ensemble = confusion_matrix(y_test, y_predict_ensemble)

# Compute the F1 score
f1_ensemble = f1_score(y_test, y_predict_ensemble, average='micro')

# Compute the accuracy score
score_ensemble = ensemble.score(x_test, y_test)

# Save the model
with open('twitter_predictions.pkl', 'wb') as file:
    pickle.dump(ensemble, file)
    
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(TfidV, file)