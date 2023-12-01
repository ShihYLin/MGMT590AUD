import pandas as pd

# Load the Excel file into a pandas DataFrame
file_path = '/Users/ziyun/Documents/MGMT590AUD/Assignment 3.xlsx'
data = pd.read_excel(file_path)

# Filter the data according to the specified conditions
restaurant_data = data['review'][:400]
print(restaurant_data.head())
movie_data = data['review'][500:900]
print(movie_data.head())

# Concatenate the filtered restaurant and movie data
training_data = pd.concat([restaurant_data, movie_data])

# Get the remaining rows as test data
test_data = pd.concat([data['review'][400:500], data['review'][900:1000]])

# Display the shapes of training and test datasets
print("Training Data Shape:", training_data.shape)
print("Test Data Shape:", test_data.shape)