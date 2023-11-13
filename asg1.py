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

# TF_IDF
from sklearn.feature_extraction.text import TfidfVectorizer
review_list_tokens = []
stop_words = set(stopwords.words('english'))

for row in df['tokenized']:
    lemmartize = [lemmatizer.lemmatize(i.lower()) for i in row if i.isalpha() and i.lower() not in stop_words]
    review_list_tokens.append(" ".join(lemmartize))
print(review_list_tokens)

# Remove word frequency < 3, TF-IDF
vectorizer3 = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
vectorizer3.fit(review_list_tokens)
v3 = vectorizer3.transform(review_list_tokens)

rows_v3, columns_v3 = v3.shape
print(f"Row #: {rows_v3}")
print(f"Column #: {columns_v3}")

# Convert the TF-IDF matrix to a dense array
tfidf_array = v3.toarray()

# Get the feature names (words)
feature_names = vectorizer3.get_feature_names_out()

# Create a DataFrame with TF-IDF vectors
tfidf_df = pd.DataFrame(tfidf_array, columns=feature_names)

# Save the DataFrame to a CSV file
tfidf_df.to_csv('/Users/ziyun/Documents/MGMT590AUD/tfidf_vectors.csv', index=False)


# POS tag, TFIDF vectorization, frequency >= 4
#Flatten the list
flat_list = [word for sublist in df['tokenized'] for word in sublist]

POS_token = nltk.pos_tag(flat_list)
print(POS_token)

# Concatenate word and POS tag in a list
POS_list = [word + pos_tag for word, pos_tag in POS_token]

# Join the words and POS tags into a single string
POS_string = " ".join(POS_list)

# Create a list with a single element (your POS_string)
documents = [POS_string]

vectorizer4 = TfidfVectorizer()
vectorizer4.fit(documents)
print(vectorizer4.vocabulary_)

#remove smaller than 4
vectorizer5 = CountVectorizer(min_df=4)

vectorizer5.fit(documents)
print(vectorizer5.vocabulary_)
