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
