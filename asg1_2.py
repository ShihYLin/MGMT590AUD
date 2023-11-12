import pandas as pd
import nltk

# Import csv file
df = pd.read_csv("/Users/ziyun/Documents/MGMT590AUD/Assignment 1.csv")

# Tokenize step 1
df['tokenized'] = df.iloc[:,1].apply(lambda x: nltk.word_tokenize(x))

print(df['tokenized'].head())