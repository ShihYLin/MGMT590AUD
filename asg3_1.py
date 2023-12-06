import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Load the Excel file into a pandas DataFrame
file_path = '/Users/ziyun/Documents/MGMT590AUD/Assignment 3.xlsx'
data = pd.read_excel(file_path)

# Filter the data according to the specified conditions
restaurant_data = data[:400]
# print(restaurant_data.head())
movie_data = data[500:900]
# print(movie_data.head())

# Concatenate the filtered restaurant and movie data
training_data = pd.concat([restaurant_data, movie_data])

# Get the remaining rows as test data
test_data = pd.concat([data[400:500], data[900:1000]])

# Display the shapes of training and test datasets
# print("Training Data Shape:", training_data.shape)
# print("Test Data Shape:", test_data.shape)

# Function to perform lemmatization and preprocessing of text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuations
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

# Apply preprocessing to the 'review' column
data['processed_review'] = data['review'].apply(preprocess_text)
print(data['processed_review'].head())

# TF-IDF Vectorizer with specified settings
tfidf_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2))

# Fit and transform the processed reviews into TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_review'])

# Get feature names (terms) from TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Display the TF-IDF matrix DataFrame
print("TF-IDF Matrix Shape:", tfidf_df.shape)
print(tfidf_df.head())

# Split the data into features (TF-IDF matrix) and target variable
X = tfidf_matrix
y = data['label']  # Replace 'label_column' with the column containing the labels

# Split the data into training and test sets (using the previously defined indices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naïve Bayes classifier (MultinomialNB)
nb_classifier = MultinomialNB()

# Train the classifier using the training data
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Initialize the Logistic Regression model
logit_model = LogisticRegression(max_iter=1000)

# Train the model using the training data
logit_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logit_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Logit Accuracy:", accuracy)
