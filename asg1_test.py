import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder

# Import csv file
df = pd.read_csv("/Users/ziyun/Documents/MGMT590AUD/Assignment 1.csv")

# Tokenize step 1
df['tokenized'] = df.iloc[:, 1].apply(lambda x: nltk.word_tokenize(x))

# Get first ten rows after tokenization
first_ten_rows = df['tokenized'][:10]
print(first_ten_rows)

# Use LabelEncoder for encoding the text data
label_encoder = LabelEncoder()
df['encoded_text'] = first_ten_rows.apply(lambda x: label_encoder.fit_transform(x))

# Specify the text file path
txt_file_path = '/Users/ziyun/Documents/MGMT590AUD/encoded_text_output.txt'

# Open the text file in write mode
with open(txt_file_path, 'w') as txtfile:
    # Write each encoded row to a new line in the text file
    for encoded_row in df['encoded_text'][:10]:
        txtfile.write(str(encoded_row) + '\n')

print(f"The encoded text has been saved to {txt_file_path}")

# Print the output dimension
print("Output Dimension:", df['encoded_text'][:10].shape)
