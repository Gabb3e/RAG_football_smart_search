from search import search
from response_generator import generate_response
from model_training import load_model_and_tokenizer, prepare_data, tokenize_data, train_model, device

# Full pipeline function: search and generate response
def query_retriever(user_query, synonym_dict, model, tokenizer, device):
    # Step 1: Search for documents based on user query
    retrieved_doc = search(user_query, synonym_dict)
    
    # Step 2: Generate a response based on the retrieved document
    return generate_response(retrieved_doc, user_query, model, tokenizer, device)

def main(device):
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Prepare the data for training
    train_dataset, eval_dataset = prepare_data('csv/simple_squad.csv', 'csv/players_squad_format.csv')

    # Tokenize the data
    tokenized_train_dataset = tokenize_data(tokenizer, train_dataset)
    tokenized_eval_dataset = tokenize_data(tokenizer, eval_dataset)

    # Train the model
    train_model(model, tokenizer, tokenized_train_dataset, tokenized_eval_dataset)

    # Test query after training
    user_query = "For which club did Miroslav Klose last play?"
    # Dictionary of football-related synonyms
    synonym_dict = {
        "goalie": "goalkeeper",
        "keeper": "goalkeeper",
        "valuation": "market value",
        "price": "market value",
        "contract end": "contract expiration",
        "worth": "market value",
        "tallest": "highest",
        "most expensive": "highest market value",
        "expensive": "market value",
        }
    # Retrieve the response
    response = query_retriever(user_query, synonym_dict, model, tokenizer, device)
    print(f"Q: {user_query}")
    print(f"A: {response}")

if __name__ == "__main__":
    main(device)
