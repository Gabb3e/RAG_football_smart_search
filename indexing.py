import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from data_preprocessing import df

# Define schema (fields to be indexed)
schema = Schema(player_id=ID(stored=True), name=TEXT(stored=True), content=TEXT(stored=True))

# Use the directory for writing the index
index_dir = "/home/g2de/documents/programmering/projects/transfer_hub/football_index"

# Ensure the directory exists
if not os.path.exists(index_dir):
    os.mkdir(index_dir)

# Create index if not exists
if not os.path.exists(os.path.join(index_dir, 'index')):
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    
    for index, row in df.iterrows():
        full_name = f"{row['first_name']} {row['last_name']}"
        content = f"Player ID: {row['player_id']}, Name: {full_name}, " \
              f"Last Season: {row['last_season']}, Current Club ID: {row['current_club_id']}, " \
              f"Player Code: {row['player_code']}, Country of Birth: {row['country_of_birth']}, City of Birth: {row['city_of_birth']}, " \
              f"Country of Citizenship: {row['country_of_citizenship']}, Date of Birth: {row['date_of_birth']}, " \
              f"Sub Position: {row['sub_position']}, Position: {row['position']}, Foot: {row['foot']}, " \
              f"Height: {row['height_in_cm']} cm, Contract Expiration: {row['contract_expiration_date']}, " \
              f"Agent: {row['agent_name']}, Image URL: {row['image_url']}, Profile URL: {row['url']}, " \
              f"Domestic Competition ID: {row['current_club_domestic_competition_id']}, " \
              f"Club: {row['current_club_name']}, Market Value: {row['market_value_in_eur']} EUR, " \
              f"Highest Market Value: {row['highest_market_value_in_eur']} EUR."
        
        # Add document to the index
        writer.add_document(player_id=str(row['player_id']),
                            name=full_name,
                            content=content)
    writer.commit()

print("index ok")