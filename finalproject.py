import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the CSV file
data = pd.read_csv('/Users/ziyun/Documents/MGMT590AUD/labeled_data_clean.csv')

# Splitting the data based on labels
false_data = data[:][:3303]
true_data = data[:][3303:6190]

# Selecting training and testing data
train_false, test_false = train_test_split(false_data, test_size=867, random_state=42)
train_true, test_true = train_test_split(true_data, test_size=867, random_state=42)

# Concatenating the training and testing data
train_data = pd.concat([train_false[:2020], train_true[:2020]])
test_data = pd.concat([test_false, test_true])

# Shuffle the training data
train_data = train_data.sample(frac=1).reset_index(drop=True)

# Tokenizing the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['content'].astype(str)) # Convert to string explicitly

# Converting text to sequences
X_train = tokenizer.texts_to_sequences(train_data['content'].astype(str))
X_test = tokenizer.texts_to_sequences(test_data['content'].astype(str))

# Padding sequences
maxlen = 100  # Define the maximum sequence length
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')

# Encoding labels
y_train = pd.get_dummies(train_data['label_bi']).values
y_test = pd.get_dummies(test_data['label_bi']).values

# Build LSTM model
embedding_dim = 128

model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, embedding_dim, input_length=maxlen))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 5

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
