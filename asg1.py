import pandas as pd
import numpy as np
import sklearn
import nltk

df = pd.read_csv("/Users/ziyun/Documents/MGMT 590 AUD/Assignment 1.csv")

df['tokenized'] = df.iloc[:,1].apply(lambda x: nltk.word_tokenize(x))

print(df.head())