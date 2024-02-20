import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("/Users/ziyun/Documents/Research/Experiment/Palampur_Leaf_Blast_1984_2004.csv")

# Set the font size for grid labels globally
plt.rc('axes', labelsize=20)

# Plotting the data
plt.figure(figsize=(50, 30))  # Adjust the figure size if needed
plt.plot(df['Order'], df['Pest Value'], marker='o', linestyle='-', color='b', label='Pest Value')
plt.plot(df['Order'], df['Maximum Temparature(°C)'], marker='o', linestyle='-', color='r', label='Max Temp')
plt.plot(df['Order'], df['Minimum Temparature(°C)'], marker='o', linestyle='-', color='g', label='Min Temp')
plt.title('Palampur Pest Value vs. Maximum and Minimum Temperature', fontsize=24)
plt.xlabel('Order', fontsize=20)
plt.ylabel('Pest Value', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('/Users/ziyun/Documents/Research/Experiment/Palampur_Disease_Temp.png')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("/Users/ziyun/Documents/Research/Experiment/Rajendranagar_Rice_blast_Data_1984_2003.csv")

# Create a new column with consecutive numbers starting from 1
df['NewColumn'] = range(1, len(df) + 1)

# Set the font size for grid labels globally
plt.rc('axes', labelsize=20)

# Plotting the data
plt.figure(figsize=(50, 30))  # Adjust the figure size if needed
plt.plot(df['NewColumn'], df['Pest Value'], marker='o', linestyle='-', color='b', label='Pest Value')
plt.plot(df['NewColumn'], df['Maximum Temparature(°C)'], marker='o', linestyle='-', color='r', label='Max Temp')
plt.plot(df['NewColumn'], df['Minimum Temparature(°C)'], marker='o', linestyle='-', color='g', label='Min Temp')
plt.title('Rajendranagar Pest Value vs. Maximum and Minimum Temperature', fontsize=24)
plt.xlabel('Order', fontsize=20)
plt.ylabel('Pest Value', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('/Users/ziyun/Documents/Research/Experiment/Rajendranagar_Disease_Temp.png')

plt.show()
