import pandas as pd

players_file_path = 'csv/players.csv'   # Load the dataset from the file path
players_df = pd.read_csv(players_file_path) # Read the CSV file into a pandas DataFrame

players_squad_format_path = 'csv/players_squad_format.csv'  # Load the query-answer dataset from the file path
qa_df = pd.read_csv(players_squad_format_path)

# List of unwanted words to remove
unwanted_words = ['advertisement', 'comments', 'tags', 'facebook', 'twitter', 'instagram', 'advertisements']

# Preprocess players data
def preprocess_players_data(df):
    """
    Preprocess the player data by cleaning unwanted text, handling missing values, 
    and converting columns to appropriate types.
    """
    df['name'] = df['name'].str.lower()  # Convert text to lowercase
    
    for word in unwanted_words:
        df['name'] = df['name'].str.replace(word, '', case=False)
    df['name'] = df['name'].str.strip() # Remove extra spaces created by the removal of unwanted words
    df['name'] = df['name'].str.replace(' +', ' ')  # Replace multiple spaces with a single space

    # Drop rows with missing important values
    df_cleaned = df.dropna(subset=['first_name', 'foot', 'height_in_cm'])
    df_cleaned = players_df.drop(columns=['image_url', 'url'])
    
    # Convert market value and height to numeric, coercing errors
    df_cleaned['market_value_in_eur'] = pd.to_numeric(df_cleaned['market_value_in_eur'], errors='coerce')
    df_cleaned['height_in_cm'] = pd.to_numeric(df_cleaned['height_in_cm'], errors='coerce')
    df_cleaned['highest_market_value_in_eur'] = pd.to_numeric(df_cleaned['highest_market_value_in_eur'], errors='coerce')
    df_cleaned.replace(to_replace="advertisement", value="", regex=True, inplace=True)
    
    # Remove rows with any remaining nulls in critical fields
    df_cleaned = df_cleaned.dropna(subset=['market_value_in_eur', 'height_in_cm', 'highest_market_value_in_eur'])

    return df_cleaned

# Preprocess query-answer data
def preprocess_players_squad_format_data(df):
    """
    Preprocess the query-answer data by creating context, calculating lengths, 
    and cleaning up text.
    """
   # Ensure the 'query' and 'answer' columns exist and are not null
    if 'query' in df.columns and 'answer' in df.columns:
        df['query'] = df['query'].str.lower().str.strip()  # Lowercase and clean query text
        df['answer'] = df['answer'].str.lower().str.strip()  # Lowercase and clean answer text

        # Create 'context' by combining the query and answer if it's not already present
        df['context'] = df['query'] + " The answer is: " + df['answer']
    
    # Perform query-answer specific preprocessing (calculating lengths)
    if 'query' in df.columns:
        df["question_length"] = df["query"].apply(len)
    if 'answer' in df.columns:
        df["answer_length"] = df["answer"].apply(len)
    if 'context' in df.columns:
        df["context_length"] = df["context"].apply(len)

    # Remove any rows where answers are not present in the context
    df = df[df.apply(lambda row: row['answer'] in row['context'], axis=1)]

    # Drop duplicates and ensure there are no empty rows
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['query', 'answer', 'context'], inplace=True)

    return df

# Run preprocessing for both datasets
def preprocess_data(players_file_path, players_squad_format_path):
    """
    Main function to preprocess both player and query-answer datasets.
    """
    # Load the datasets
    players_df = pd.read_csv(players_file_path)
    qa_df = pd.read_csv(players_squad_format_path)

    # Clean the player data
    cleaned_players_df = preprocess_players_data(players_df)
    
    # Clean the query-answer data
    cleaned_qa_df = preprocess_players_squad_format_data(qa_df)
    
    # Save the cleaned datasets back to CSV (optional)
    cleaned_players_df.to_csv('csv/cleaned_players.csv', index=False)
    cleaned_qa_df.to_csv('csv/cleaned_qa_data.csv', index=False)
    
    print("Data cleaning completed!")
    return cleaned_players_df, cleaned_qa_df

cleaned_players_df, cleaned_qa_df = preprocess_data(players_file_path, players_squad_format_path)

#print(cleaned_players_df.head())
#print(cleaned_players_df.columns)
#print(cleaned_qa_df.head())
#print(cleaned_qa_df.columns)
#print(df_cleaned.info())
