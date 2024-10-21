from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
from evaluate import load
import torch.nn as nn
import pandas as pd
import torch

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the evaluation metric
metric_f1 = load("f1")
metric_exact_match = load("exact_match")

def load_model_and_tokenizer(model_path='facebook/bart-large', tokenizer_path='facebook/bart-large'):
    # Load pre-trained BART model and tokenizer
    model = BartForConditionalGeneration.from_pretrained(model_path, ignore_mismatched_sizes=True)

    # Ensure the encoder and decoder token embeddings are properly initialized
    if model.config.tie_word_embeddings:
        model.model.encoder.embed_tokens = model.model.decoder.embed_tokens  # Sharing the embeddings between encoder and decoder
    else:
        # If they are not shared, initialize both embeddings
        model.model.encoder.embed_tokens.weight = nn.Parameter(model.model.decoder.embed_tokens.weight.clone())

    model.lm_head = nn.Linear(model.config.d_model, model.config.vocab_size)

    for param in model.parameters(): # Freeze all layers in the model except the output head (lm_head)
        param.requires_grad = False  # Freeze all parameters
    for param in model.lm_head.parameters(): # Unfreeze only the last layer (output head) for fine-tuning
        param.requires_grad = True  # Unfreeze the lm_head (output head)
    for param in model.model.decoder.layers[-6:].parameters(): # Unfreeze only the last few layers of the decoder for fine-tuning
        param.requires_grad = True  # Unfreeze the last X layers of the decoder

    model.to(device)  # Move the model to the device (GPU if available)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer

def prepare_data(squad_df, players_df):
    # Load the Simple SQuAD dataset
    df_squad = pd.read_csv(squad_df)
    #print(df_squad.head())
    #print(df_squad.columns)
    #print(df_squad.shape)
    df_players = pd.read_csv(players_df)
    #print(df_players.head())
    #print(df_players.columns)
    #print(df_players.shape)

    df_combined = pd.concat([df_squad, df_players], ignore_index=True)

    # Split the dataset into training and evaluation sets
    train_df, eval_df = train_test_split(df_combined, test_size=0.2)  # 80% train, 20% eval
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    return train_dataset, eval_dataset

# Tokenize the inputs and outputs
def tokenize_data(tokenizer, dataset):
    def tokenize_function(examples):
        # Ensure all inputs are strings and not None
        context = str(examples['context']) if examples['context'] is not None else ""
        question = str(examples['question']) if examples['question'] is not None else ""
        answer = str(examples['answer']) if examples['answer'] is not None else ""
        
        model_inputs = tokenizer(
            context,
            question,
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        # Tokenize 'answer' (targets) and ensure that it is in string format
        if isinstance(examples['answer'], list):
            labels = tokenizer(
                text_target=answer,
                max_length=150,
                truncation=True,
                padding="max_length"
            )
        else:
            labels = tokenizer(
                text_target=str(examples['answer']),
                max_length=150,
                truncation=True,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Apply the tokenization to training and evaluation datasets
    tokenized_train_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_train_dataset.set_format("torch")

    # Convert the datasets to PyTorch format
    #tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
    #tokenized_eval_dataset.set_format("torch")

    return tokenized_train_dataset

def train_model(model, tokenizer, train_dataset, eval_dataset):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results_squad",
        eval_strategy="epoch",
        save_strategy="epoch",  
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        dataloader_num_workers=4,
        num_train_epochs=1,
        weight_decay=0.1,
        save_total_limit=2,
        save_steps=500,
        gradient_accumulation_steps=3,
        greater_is_better=False, 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )
    # Start the fine-tuning
    trainer.train()

    # Save the fine-tuned model for future use
    model.save_pretrained("./fine_tuned_bart")
    tokenizer.save_pretrained("./fine_tuned_bart")

# Custom compute_metrics function
def compute_metrics(eval_pred, tokenizer):
    # Function to post-process model outputs to calculate metrics
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels
    
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Post-process predictions and labels
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    # Exact Match (EM)
    exact_match_score = metric_exact_match.compute(predictions=decoded_preds, references=decoded_labels)
    
    # F1 Score
    f1_score = metric_f1.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "exact_match": exact_match_score['exact_match'],
        "f1": f1_score['f1']
    }

# Main execution function
def main():
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Prepare the data
    train_dataset, eval_dataset = prepare_data('csv/simple_squad.csv', 'csv/players.csv')

    # Tokenize the data
    tokenized_train_dataset = tokenize_data(tokenizer, train_dataset)
    tokenized_eval_dataset = tokenize_data(tokenizer, eval_dataset)

    # Train the model
    train_model(model, tokenizer, tokenized_train_dataset, tokenized_eval_dataset)


# Execute the main function
if __name__ == "__main__":
    main()
