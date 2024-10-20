import pandas as pd

players_file_path = 'csv/players.csv'   # Load the dataset from the file path
players_df = pd.read_csv(players_file_path) # Read the CSV file into a pandas DataFrame

qa_file_path = 'csv/qa_data.csv'  # Load the query-answer dataset from the file path
qa_df = pd.read_csv(qa_file_path)

# List of unwanted words to remove
unwanted_words = ['advertisement', 'comments', 'tags', 'facebook', 'twitter', 'instagram', 'advertisements']

# Preprocess players data
def preprocess_players_data(df):
    # Convert text to lowercase and remove unwanted words using pandas' replace method
    df['name'] = df['name'].str.lower()  # Convert text to lowercase
    
    for word in unwanted_words:
        df['name'] = df['name'].str.replace(word, '', case=False)
    
    # Remove extra spaces created by the removal of unwanted words
    df['name'] = df['name'].str.strip()
    df['name'] = df['name'].str.replace(' +', ' ')  # Replace multiple spaces with a single space

    # Drop rows with missing important values
    df_cleaned = df.dropna(subset=['first_name', 'foot', 'height_in_cm'])
    df_cleaned = players_df.drop(columns=['image_url', 'url'])
    
    # Convert market value and height to numeric, coercing errors
    df_cleaned['market_value_in_eur'] = pd.to_numeric(df_cleaned['market_value_in_eur'], errors='coerce')
    df_cleaned['height_in_cm'] = pd.to_numeric(df_cleaned['height_in_cm'], errors='coerce')
    df_cleaned['highest_market_value_in_eur'] = pd.to_numeric(df_cleaned['highest_market_value_in_eur'], errors='coerce')
    df_cleaned.replace(to_replace="advertisement", value="", regex=True, inplace=True)
    
    return df_cleaned

# Preprocess query-answer data
def preprocess_qa_data(df):
    # Check if the necessary columns exist
    if 'query' in df.columns and 'answer' in df.columns:
        # Create a 'context' column by combining query and answer (if it's not already present)
        df['context'] = df['query'] + " The answer is: " + df['answer']
    
    # Perform query-answer specific preprocessing (calculating lengths)
    if 'query' in df.columns:
        df["question_length"] = df["query"].apply(len)
    if 'answer' in df.columns:
        df["answer_length"] = df["answer"].apply(len)
    if 'context' in df.columns:
        df["context_length"] = df["context"].apply(len)

    return df

# Clean both datasets
cleaned_players_df = preprocess_players_data(players_df)
cleaned_qa_df = preprocess_qa_data(qa_df)

#print(cleaned_players_df.head())
print(cleaned_players_df.columns)
#print(cleaned_qa_df.head())
print(cleaned_qa_df.columns)
#print(df_cleaned.info())
