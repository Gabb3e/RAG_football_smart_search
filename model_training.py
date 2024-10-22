from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset
from evaluate import load
import pandas as pd
import torch

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the evaluation metric
metric_f1 = load("f1")
metric_exact_match = load("exact_match")

def load_model_and_tokenizer(model_path='bert-base-uncased', tokenizer_path='bert-base-uncased'):
    # Load pre-trained BART model and tokenizer
    model = BertForQuestionAnswering.from_pretrained(model_path, ignore_mismatched_sizes=True)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    model.to(device)  # Move the model to the device (GPU if available)

    return model, tokenizer

def prepare_player_qa(players_df):
    # Generate QA pairs from player information
    qa_pairs = []
    for _, row in players_df.iterrows():
        # Example: Create a question about the player's current club
        context = (
            f"{row['first_name']} {row['last_name']} is a football player born in {row['country_of_birth']} "
            f"on {row['date_of_birth']}. He currently plays for {row['current_club_name']}."
        )
        question = f"What is the current club of {row['first_name']} {row['last_name']}?"
        answer = row['current_club_name']
        
        # Find the start and end positions of the answer in the context
        start_pos = context.find(answer)
        end_pos = start_pos + len(answer) if start_pos != -1 else 0
        
        qa_pairs.append({
            'context': context,
            'question': question,
            'answer': answer,
            'context_length': len(context),
            'question_length': len(question),
            'answer_length': len(answer),
            'start_positions': start_pos,
            'end_positions': end_pos
        })
    
    # Convert the list to a DataFrame
    qa_df = pd.DataFrame(qa_pairs)
    return qa_df

def prepare_data(squad_df, players_df):
    # Load the Simple SQuAD dataset
    df_squad = pd.read_csv(squad_df)
    
    # Prepare the player data in QA format
    players_df = pd.read_csv(players_df)
    qa_df = prepare_player_qa(players_df)
    
    # Combine both datasets
    df_combined = pd.concat([df_squad, qa_df], ignore_index=True)
    
    # Compute start and end positions if not already present
    if 'start_positions' not in df_combined.columns or 'end_positions' not in df_combined.columns:
        def find_answer_positions(row):
            context = row['context']
            answer = row['answer']
            start_pos = context.find(answer)
            if start_pos == -1:
                # Handle cases where the answer is not found
                start_pos = 0
                end_pos = 0
            else:
                end_pos = start_pos + len(answer)
            return pd.Series({'start_positions': start_pos, 'end_positions': end_pos})
        
        positions = df_combined.apply(find_answer_positions, axis=1)
        df_combined = pd.concat([df_combined, positions], axis=1)
    
    # Split the dataset into training and evaluation sets
    train_df, eval_df = train_test_split(df_combined, test_size=0.2, random_state=42)  # 80% train, 20% eval
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    return train_dataset, eval_dataset

# Tokenize the inputs and outputs
def tokenize_data(tokenizer, dataset):
    def tokenize_function(examples):
        # Tokenize the inputs
        tokenized_examples = tokenizer(
            examples['context'],
            examples['question'],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Initialize start and end positions
        start_positions = []
        end_positions = []
        
        for i in range(len(examples['answer'])):
            answer = examples['answer'][i]
            start_char = examples['start_positions'][i]
            end_char = examples['end_positions'][i]
            
            # Find the start and end token indices
            offset = tokenized_examples['offset_mapping'][i]
            start_token = 0
            end_token = 0
            for idx, (start, end) in enumerate(offset):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
                    break
            start_positions.append(start_token)
            end_positions.append(end_token)
        
        tokenized_examples['start_positions'] = start_positions
        tokenized_examples['end_positions'] = end_positions
        tokenized_examples.pop('offset_mapping')  # Remove offset mapping as it's no longer needed
        
        return tokenized_examples
    
    # Apply the tokenization to the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
    
    return tokenized_dataset

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
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,
        gradient_accumulation_steps=3,
        greater_is_better=False, 
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs_qa",
        logging_steps=10,
        report_to="none",
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, eval_dataset)
    )
    # Start the fine-tuning
    trainer.train()

    # Save the fine-tuned model for future use
    model.save_pretrained("./fine_tuned_bert")
    tokenizer.save_pretrained("./fine_tuned_bert")

# Custom compute_metrics function
def compute_metrics(eval_pred, eval_dataset):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    start_preds, end_preds = predictions
    start_labels, end_labels = labels[:, 0], labels[:, 1]
    
    # Convert predictions to start and end indices
    start_preds = start_preds.argmax(axis=1)
    end_preds = end_preds.argmax(axis=1)
    
    # Initialize lists for decoded predictions and labels
    decoded_preds = []
    decoded_labels = []
    
    for i in range(len(start_preds)):
        start = start_preds[i]
        end = end_preds[i]
        context = eval_dataset[i]['context']
        if start < len(context) and end < len(context):
            answer = context[start:end]
        else:
            answer = ""
        decoded_preds.append(answer)
        decoded_labels.append(eval_dataset[i]['answer'])
    
    # Compute exact match and F1 scores
    exact_match = metric_exact_match.compute(predictions=decoded_preds, references=decoded_labels)
    f1 = metric_f1.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        "exact_match": exact_match['exact_match'],
        "f1": f1['f1']
    }

# Main execution function
def main():
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Prepare the data
    train_dataset, eval_dataset = prepare_data('csv/simple_squad.csv', 'csv/qa_data.csv')
    print("Training Sample:", train_dataset[:5])
    print("Evaluation Sample:", eval_dataset[:5])

    # Tokenize the data
    tokenized_train_dataset = tokenize_data(tokenizer, train_dataset)
    tokenized_eval_dataset = tokenize_data(tokenizer, eval_dataset)

    # Train the model
    train_model(model, tokenizer, tokenized_train_dataset, tokenized_eval_dataset)


# Execute the main function
if __name__ == "__main__":
    main()
