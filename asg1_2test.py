import pandas as pd
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import padding function
from sklearn.preprocessing import OneHotEncoder
from array import array

# Import csv file
df = pd.read_csv("/Users/ziyun/Documents/MGMT590AUD/Assignment 1.csv")

# Tokenize step 1
df['tokenized'] = df.iloc[:, 1].apply(lambda x: nltk.word_tokenize(x))

# Choose the first 10 tokenized rows
tokenized_samples = df['tokenized'][:10]

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
