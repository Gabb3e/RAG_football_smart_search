import pandas as pd

# Load the dataset from the file path
file_path = 'players.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# List of unwanted words to remove
unwanted_words = ['advertisement', 'comments', 'tags', 'facebook', 'twitter', 'instagram', 'advertisements']

def clean_text_pandas(df, column):
    # Convert text to lowercase
    df[column] = df[column].str.lower()
    
    # Remove unwanted words using pandas' replace method
    for word in unwanted_words:
        df[column] = df[column].str.replace(word, '', case=False)
    
    # Remove extra spaces created by the removal of unwanted words
    df[column] = df[column].str.strip()
    df[column] = df[column].str.replace(' +', ' ')  # Replace multiple spaces with a single space
    
    return df

df_cleaned = clean_text_pandas(df, 'name')

# 1. Handle missing values - for simplicity, let's drop rows with missing important values (like first name, foot)
df_cleaned = df.dropna(subset=['first_name', 'foot', 'height_in_cm'])
df_cleaned = df.drop(columns=['image_url', 'url'])

df_cleaned['market_value_in_eur'] = pd.to_numeric(df_cleaned['market_value_in_eur'], errors='coerce')
df_cleaned['highest_market_value_in_eur'] = pd.to_numeric(df_cleaned['highest_market_value_in_eur'], errors='coerce')
df_cleaned['height_in_cm'] = pd.to_numeric(df_cleaned['height_in_cm'], errors='coerce')

df_cleaned.replace(to_replace="advertisement", value="", regex=True, inplace=True)

# Print the first few rows of the DataFrame to check the data
#print(df.head())
print(df_cleaned.head())
print(df_cleaned.columns)
print(df_cleaned.shape)
#print(df_cleaned.info())