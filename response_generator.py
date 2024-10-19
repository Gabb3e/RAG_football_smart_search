# Function to generate a response based on the query and retrieved document
def generate_response(retrieved_doc, user_query, model, tokenizer, device):
    if not retrieved_doc:
        return "No relevant documents found."
    
    # Tokenize the input
    inputs = tokenizer(
        user_query, 
        max_length=512, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )

    # Move the inputs to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate a response
    summary_ids = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        num_beams=4,
        max_length=150, 
        min_length=50,
        early_stopping=True,
        no_repeat_ngram_size=4,
        length_penalty=2.0,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        forced_bos_token_id=tokenizer.bos_token_id
    )
    
    # Decode and return the output
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)