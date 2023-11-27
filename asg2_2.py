import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

#import csv file
df = pd.read_excel("/Users/ziyun/Documents/MGMT590AUD/Assignment 2 text.xlsx")

#tokenize and lowercase
df['tokenized'] = df.iloc[:,1].apply(lambda x: nltk.word_tokenize(x))
# print(df['tokenized'].head())

# Lemmatize the 'tokenized' column in each row and remove punctuation
lemmatizer = nltk.stem.WordNetLemmatizer()
lemmatized_tokens = []

for row in df['tokenized']:
    lemmatized_row = [lemmatizer.lemmatize(token).lower() for token in row if token.isalpha()]
    lemmatized_tokens.append(lemmatized_row)

# Print the lemmatized tokens for each row
# for tokens in lemmatized_tokens:
#     print(tokens)

# Remove stop words and punctuation
def remove_stopwords(word_list):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in word_list if word.lower() not in stop_words and word.isalpha()]
    return filtered_words

filtered_list_of_lists = [remove_stopwords(word_list) for word_list in lemmatized_tokens]
# Print the filtered lists
print("List of Lists without Stopwords:")
for word_list in filtered_list_of_lists:
    print(word_list)

# Drop frequency < 5
#Flatten the list
flat_list = [' '.join(sublist) for sublist in filtered_list_of_lists]

# Use CountVectorizer for unigrams and bigrams with minimum frequency of 5
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5)
X = vectorizer.fit_transform(flat_list)
print(vectorizer.vocabulary_)

# Convert the sparse matrix to a DataFrame representing the term-document matrix
term_document_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Display the term-document matrix
print(term_document_matrix)
