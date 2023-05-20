import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Load data
train_df = pd.read_csv('/kaggle/input/deeptweets2/train.csv')
test_df = pd.read_csv('/kaggle/input/deeptweets2/test.csv')


# Preprocessing
def preprocess(text):
    text = re.sub(r'\#\w+|\@\w+|[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

train_df['TweetText'] = train_df['TweetText'].apply(preprocess)
test_df['TweetText'] = test_df['TweetText'].apply(preprocess)

# Feature engineering
def extract_features(df):
    # Extract features from hashtags and mentions
    df['has_sport_hashtag'] = df['TweetText'].str.contains('#sport', case=False)
    df['has_politics_hashtag'] = df['TweetText'].str.contains('#politics', case=False)
    df['has_un_mention'] = df['TweetText'].str.contains('@un', case=False)
    df['has_bbc_sport_mention'] = df['TweetText'].str.contains('@bbcsport', case=False)
    df['has_president_mention'] = df['TweetText'].str.contains('president', case=False)
    
    # Spot incorrectly labeled tweets
    keywords = ['sport', 'politic', 'football']
    df['has_keyword'] = df['TweetText'].str.contains('|'.join(keywords), case=False)
    
    return df

train_df = extract_features(train_df)
test_df = extract_features(test_df)



# Splitting the data into training and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Model training and evaluation
count_vect = CountVectorizer().fit(train_data['TweetText'])

X_train_cv = count_vect.transform(train_data['TweetText'])
X_val_cv = count_vect.transform(val_data['TweetText'])

tfidf_vect = TfidfVectorizer().fit(train_data['TweetText'])

X_train_tfidf = tfidf_vect.transform(train_data['TweetText'])
X_val_tfidf = tfidf_vect.transform(val_data['TweetText'])

y_train = train_data['Label']
y_val = val_data['Label']

# Training a Naive Bayes classifier using CountVectorizer
nb_cv = MultinomialNB()
nb_cv.fit(X_train_cv, y_train)

# Evaluating the Naive Bayes classifier using CountVectorizer
y_pred_cv = nb_cv.predict(X_val_cv)
print("Naive Bayes classifier with CountVectorizer")
print(classification_report(y_val, y_pred_cv))

# Training a Naive Bayes classifier using TfidfVectorizer
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

# Evaluating the Naive Bayes classifier using TfidfVectorizer
y_pred_tfidf = nb_tfidf.predict(X_val_tfidf)
print("Naive Bayes classifier with TfidfVectorizer")
print(classification_report(y_val, y_pred_tfidf))
