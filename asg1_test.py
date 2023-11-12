import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder

# Import csv file
df = pd.read_csv("/Users/ziyun/Documents/MGMT590AUD/Assignment 1.csv")

# Tokenize step 1
df['tokenized'] = df.iloc[:, 1].apply(lambda x: nltk.word_tokenize(x))

# Get first ten rows after tokenization
first_ten_rows = df['tokenized'].head(10)
print(first_ten_rows)

# Use LabelEncoder for encoding the text data
label_encoder = LabelEncoder()
df['encoded_text'] = first_ten_rows.apply(lambda x: label_encoder.fit_transform(x))

# Save the encoded text to a text file
df['encoded_text'].to_csv("/Users/ziyun/Documents/MGMT590AUD/encoded_text.txt", header=False, index=False)
