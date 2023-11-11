import pandas as pd
import numpy as np
import sklearn
import nltk

#import csv file
df = pd.read_csv("/Users/ziyun/Documents/MGMT590AUD/Assignment 1.csv")

#tokenize and lowercase
df['tokenized'] = df.iloc[:,1].apply(lambda x: nltk.word_tokenize(x))

print(df['tokenized'].head())

# Lemmatize the 'tokenized' column in each row and remove punctuation
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatized_tokens = []

for row in df['tokenized']:
    lemmatized_row = [lemmatizer.lemmatize(token) for token in row if token.isalpha()]
    lemmatized_tokens.append(lemmatized_row)

# Print the lemmatized tokens for each row
for tokens in lemmatized_tokens:
    print(tokens)
    
# Remove stop words and punctuation
from nltk.corpus import stopwords
def remove_stopwords(word_list):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in word_list if word.lower() not in stop_words and word.isalpha()]
    return filtered_words

filtered_list_of_lists = [remove_stopwords(word_list) for word_list in lemmatized_tokens]
# Print the filtered lists
print("List of Lists without Stopwords:")
for word_list in filtered_list_of_lists:
    print(word_list)

# Drop frequency < 3 words
from sklearn.feature_extraction.text import CountVectorizer

#Flatten the list
flat_list = [word for sublist in filtered_list_of_lists for word in sublist]

vectorizer1 = CountVectorizer(min_df=3)
vectorizer1.fit(flat_list)
print(vectorizer1.vocabulary_)

#TF_IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer()
vectorizer2.fit(flat_list)
print(vectorizer2.vocabulary_)

v2 = vectorizer2.transform(flat_list)
print(v2.toarray())
