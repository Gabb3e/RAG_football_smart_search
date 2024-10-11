from model_training import model, tokenizer

# Function to generate a response based on the query and retrieved document
def generate_response(retrieved_doc, user_query):
    if not retrieved_doc:
        return "No relevant documents found."
    
    # Combine the query and the retrieved document
    input_text = user_query + " " + retrieved_doc

    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate a response
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    
    # Decode and return the output
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)