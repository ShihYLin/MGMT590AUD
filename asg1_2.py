import pandas as pd
import nltk
from numpy import array
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Import csv file
df = pd.read_csv("/Users/ziyun/Documents/MGMT590AUD/Assignment 1.csv")

# Tokenize step 1
df['tokenized'] = df.iloc[:,1].apply(lambda x: nltk.word_tokenize(x))

# Get first ten rows
first_ten_rows = df['tokenized'].head(10)
print(first_ten_rows)

# Flatten the list
flat_list = [word for sublist in first_ten_rows for word in sublist]

# Word count
count_words = Counter(flat_list).most_common()

# Define a vocabulary
vocab_index = {w: i + 1 for i, (w, c) in enumerate(count_words)}

# Encoding
index_encoded_matrix = []
for row in first_ten_rows:
    index_encoded_row = [vocab_index[word] for word in row]
    index_encoded_matrix.append(index_encoded_row)

# Convert the list of lists to a 2D array (matrix)
index_encoded_matrix = pd.DataFrame(index_encoded_matrix).to_numpy()

# Save the matrix to a CSV file
pd.DataFrame(index_encoded_matrix).to_csv('/Users/ziyun/Documents/MGMT590AUD/index_encoded_matrix.csv', index=False)
print("Index-encoded matrix saved to index_encoded_matrix.csv")
# Print the dimensions of the final matrix
print("Dimensions of the final matrix:", index_encoded_matrix.shape)
