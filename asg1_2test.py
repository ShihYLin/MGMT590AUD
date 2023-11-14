import pandas as pd
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import padding function
from sklearn.preprocessing import OneHotEncoder
from array import array
from gensim.models import KeyedVectors

# Import csv file
df = pd.read_csv("/Users/ziyun/Documents/MGMT590AUD/Assignment 1.csv")

# Tokenize step 1
df['tokenized'] = df.iloc[:, 1].apply(lambda x: nltk.word_tokenize(x))

# Choose the first 10 tokenized rows
tokenized_samples = df['tokenized'][:10]

# A set for all possible words
words = [word for sublist in tokenized_samples for word in sublist]

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode each word and represent each row in a vector of indices
encoded_samples = []
for sample in tokenized_samples:
    encoded_sample = label_encoder.fit_transform(sample)
    encoded_samples.append(encoded_sample)

# Pad sequences to have the same length
padded_sequences = pad_sequences(encoded_samples, padding='post')

# Save the representation of the whole collection as a 2D array
encoded_matrix = np.array(padded_sequences)

# Save the output as a txt file
np.savetxt('/Users/ziyun/Documents/MGMT590AUD/encoded_matrix.txt', encoded_matrix, fmt='%d')

# Print the output dimension
print("Output Dimension:", encoded_matrix.shape)

# One-hot encoding
# vocabulary
indices = [j for i in encoded_matrix for j in i]

# Convert indices to a numpy array of strings
indices_array = np.array(indices).reshape(-1, 1)

# No need for astype(str) in this case
indices_list2_flatten = indices_array.flatten().tolist()

# Use OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoder = onehot_encoder.fit(indices_array)

# encoding
onehot_encoded1 = [onehot_encoder.transform([[i] for i in doc_i]).tolist() for doc_i in encoded_matrix]
onehot_encoded2 = [onehot_encoder.transform(doc_i.reshape(-1, 1)).tolist() for doc_i in
encoded_matrix]

# from token to onehot
# vocabulary
words_list = [[i] for i in words]
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoder = onehot_encoder.fit(words_list)

# encoding
onehot_encoded3 = [onehot_encoder.transform([[word] for word in doc]) for doc in tokenized_samples]
print(onehot_encoded3)

# Print the dimensions of each array in the list
for i, encoded_array in enumerate(onehot_encoded3):
    print(f"Array {i + 1} Dimension:", encoded_array.shape)

# Specify the text file path
txt_file_path = '/Users/ziyun/Documents/MGMT590AUD/onehot_encoded_output.txt'

# Vertically stack the arrays
stacked_array = np.vstack(onehot_encoded3)

# Save the vertically stacked array to a txt file
np.savetxt(txt_file_path, stacked_array, fmt='%d')

print(f"The stacked onehot encoded arrays have been saved to {txt_file_path}")

# 3 GloVe
# Pre-trained glove
glove_model_path = '/Users/ziyun/Documents/MGMT590AUD/glove.6B/glove.6B.50d.txt'

# Convert GloVe model to Word2Vec format
word2vec_output_file = 'glove.6B.50d.txt.word2vec'
# glove2word2vec(glove_model_path, word2vec_output_file)

filename = 'glove.6B.50d.txt.word2vec'
glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)

# Define the dimensions of the embeddings
embedding_dim = 50
max_token = max(len(doc) for doc in tokenized_samples)

# Initialize an array to store the embeddings
embedding_matrix = np.zeros((len(tokenized_samples), max_token, embedding_dim))

# Iterate over each tokenized sample and embed each word
for i, sample in enumerate(tokenized_samples):
    for j, word in enumerate(sample):
        try:
            word_embedding = glove_model[word]
            embedding_matrix[i, j, :] = word_embedding
        except KeyError:
            pass

# Save the embedding matrix as a 3D array
np.save('embedding_matrix.npy', embedding_matrix)

# Save the 3D array to a text file
np.savetxt('/Users/ziyun/Documents/MGMT590AUD/embedding_matrix.txt', embedding_matrix.reshape(embedding_matrix.shape[0], -1), delimiter=' ')

# Print the shape of the resulting 3D array
print("Shape of the 3D array:", embedding_matrix.shape)
