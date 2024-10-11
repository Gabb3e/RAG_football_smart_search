import pandas as pd

# Load the dataset from the file path
file_path = '/home/g2de/documents/programmering/csv/fotboll_test_csv/players.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Print the first few rows of the DataFrame to check the data
#print(df.head())
#print(df.columns)