from search import search
from response_generator import generate_response
from model_training import model, tokenizer, device

# Full pipeline function: search and generate response
def query_retriever(user_query):
    # Step 1: Search for documents based on user query
    retrieved_doc = search(user_query)
    
    # Step 2: Generate a response based on the retrieved document
    return generate_response(retrieved_doc, user_query, model, tokenizer, device)

# Example usage
user_query = "Who are the players from Barcelona worth more than 60 million?"
response = query_retriever(user_query)
print(response)

