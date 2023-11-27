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

# Display the topic distribution for the first 10 rows
print("Topic distribution for the first 10 restaurant reviews:")
for i, row in enumerate(topic_results[:10]):
    print(f"Row {i+1}: {row}")

# Function to get the top-2 topics for a given row
def get_top_topics(row, num_top_topics):
    top_topics = row.argsort()[-num_top_topics:][::-1]
    return top_topics

# Get the top topics for the first 10 rows
num_top_topics = 1
print("\nTop topics for the first 10 restaurant reviews:")
for i, row in enumerate(topic_results[:10]):
    top_topics = get_top_topics(row, num_top_topics)
    print(f"ID {i+1}: Top topic - {top_topics+1}")

# Display the topic distribution for the first 10 movie reviews
print("Topic distribution for the first 10 movie reviews:")
for i, row in enumerate(topic_results[500:510]):
    print(f"Row {i+1}: {row}")

# Get the top topics for id 501 to 510
num_top_topics = 1
print("\nTop topics for movie reviews 501 to 510:")
for i, row in enumerate(topic_results[500:510], start=500+1):
    top_topics = get_top_topics(row, num_top_topics)
    print(f"ID {i}: Top topics - {top_topics+1}")
