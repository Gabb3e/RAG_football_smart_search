import pandas as pd

# Load the dataset from the file path
file_path = 'players.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)
df_cleaned = df.drop(columns=['image_url', 'url'])

# Print the first few rows of the DataFrame to check the data
#print(df.head())
#print(df_cleaned.head())
print(df_cleaned.columns)
print(df_cleaned.shape)
#print(df_cleaned.info())