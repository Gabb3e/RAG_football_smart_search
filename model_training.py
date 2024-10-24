from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from datasets import Dataset
from evaluate import load
import torch.nn as nn
import pandas as pd
import torch
import os

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the evaluation metric
metric_f1 = load("f1")
metric_exact_match = load("exact_match")

def load_model_and_tokenizer(model_path='bert-base-uncased', tokenizer_path='bert-base-uncased'):
    # Load pre-trained BART model and tokenizer
    model = BertForQuestionAnswering.from_pretrained(model_path, ignore_mismatched_sizes=True)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    model.to(device)  # Move the model to the device (GPU if available)
    # Wrap model in DistributedDataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')  # Multi-GPU backend
        model = DDP(model)
    else:
        print("Single GPU detected. Skipping DDP initialization.")

    return model, tokenizer

# Preprocessing example function (ensure at least 1 dimension for scalars)
def preprocess_output(output):
    if output.ndim == 0:
        output = output.unsqueeze(0)  # Ensure output is at least 1-dimensional
    return output

# Example of a loss function that returns a tensor, not a scalar
def compute_loss(outputs, labels):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    return preprocess_output(loss)  # Unsqueeze if necessary

def prepare_player_qa(players_squad_df):
    # Generate QA pairs from the players_squad_format.csv
    qa_pairs = []
    for _, row in players_squad_df.iterrows():
        context = row['context']
        question = row['question']
        answer = row['answer']
        
        # Ensure context, question, and answer are valid strings
        context = str(context) if pd.notnull(context) else ""
        question = str(question) if pd.notnull(question) else ""
        answer = str(answer) if pd.notnull(answer) else ""
        
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

def find_answer_positions(row):
    context = row['context']
    answer = row['answer']

    # Ensure context and answer are strings, replace NaN or None with empty strings
    if not isinstance(context, str):
        context = ""
    if not isinstance(answer, str):
        answer = ""

    start_pos = context.find(answer)
    if start_pos == -1:
        # Handle cases where the answer is not found
        start_pos = 0
        end_pos = 0
    else:
        end_pos = start_pos + len(answer)
    return pd.Series({'start_positions': start_pos, 'end_positions': end_pos})
               
def prepare_data(squad_df, players_squad_format_df):
    # Load the Simple SQuAD dataset
    df_squad = pd.read_csv(squad_df)
    
    # Prepare the player data in QA format
    df_players_squad = pd.read_csv(players_squad_format_df)
    df_players_squad = prepare_player_qa(df_players_squad)
    
    # Combine both datasets
    df_combined = pd.concat([df_squad, df_players_squad], ignore_index=True)

    # Ensure context and answer columns are valid strings and handle missing values
    df_combined['context'] = df_combined['context'].fillna("").astype(str)
    df_combined['answer'] = df_combined['answer'].fillna("").astype(str)
    
        # Avoid adding duplicate columns
    if 'start_positions' not in df_combined.columns or 'end_positions' not in df_combined.columns:
        # Compute start and end positions
        positions = df_combined.apply(find_answer_positions, axis=1)
        df_combined = pd.concat([df_combined, positions], axis=1)

    # Split the dataset into training and evaluation sets
    train_df, eval_df = train_test_split(df_combined, test_size=0.2, random_state=42)  # 80% train, 20% eval
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    return train_dataset, eval_dataset

# Custom compute_metrics function
def compute_metrics(eval_pred):
    start_preds, end_preds = eval_pred.predictions
    start_labels, end_labels = eval_pred.label_ids

    # Convert start and end predictions to max indices
    start_preds = start_preds.argmax(axis=1)
    end_preds = end_preds.argmax(axis=1)

    # Initialize lists for decoded predictions and labels
    decoded_preds = []
    decoded_labels = []

    for i in range(len(start_preds)):
        # Create an answer span using the predicted start and end
        pred_answer = f"{start_preds[i]}-{end_preds[i]}"
        true_answer = f"{start_labels[i]}-{end_labels[i]}"
        
        decoded_preds.append(pred_answer)
        decoded_labels.append(true_answer)

    # Compute Exact Match and F1 scores
    exact_match = metric_exact_match.compute(predictions=decoded_preds, references=decoded_labels)
    f1 = metric_f1.compute(predictions=decoded_preds, references=decoded_labels)

    print(f"Exact Match: {exact_match['exact_match']}, F1 Score: {f1['f1']}")

    return {
        "eval_exact_match": exact_match['exact_match'],  # Prefix with 'eval_'
        "eval_f1": f1['f1']  # Prefix with 'eval_'
    }

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
            context = examples['context'][i]
            answer = examples['answer'][i]
            
            # Ensure context and answer are strings
            if not isinstance(context, str) or not isinstance(answer, str):
                print(f"Warning: Skipping example with missing context or answer: {examples['context'][i]} or {examples['answer'][i]}")
                start_positions.append(0)  # or some default value
                end_positions.append(0)    # or some default value
                continue

            # Try to find the answer's position in the context
            start_char = context.find(answer)
            if start_char == -1:
                print(f"Warning: Missing start or end position for context: {examples['context'][i]}")
                start_positions.append(0)  # or some default value
                end_positions.append(0)    # or some default value
                continue

            end_char = start_char + len(answer)

            # Find the start and end token indices
            offset = tokenized_examples['offset_mapping'][i]
            start_token = None
            end_token = None
            for idx, (start, end) in enumerate(offset):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
                    break

            if start_token is None or end_token is None:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_positions.append(int(start_token))
                end_positions.append(int(end_token))

        return {
            'input_ids': tokenized_examples['input_ids'],
            'attention_mask': tokenized_examples['attention_mask'],
            'start_positions': start_positions,
            'end_positions': end_positions,
        }
    # Apply the tokenization to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=32, 
        num_proc=8,
        keep_in_memory=True)
    
    # Ensure that the output is a Dataset object
    if isinstance(tokenized_dataset, tuple):
        raise ValueError("The tokenization function returned a tuple instead of a Dataset object.")
    
    # Explicitly cast the 'start_positions' and 'end_positions' to int64
    from datasets import Value
    tokenized_dataset = tokenized_dataset.cast_column("start_positions", Value("int64"))
    tokenized_dataset = tokenized_dataset.cast_column("end_positions", Value("int64"))

    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
    
    return tokenized_dataset

def train_model(model, tokenizer, train_dataset, eval_dataset):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results_squad",
        eval_strategy="epoch",
        save_strategy="epoch",  
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=3,
        dataloader_num_workers=8,
        num_train_epochs=3,
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_total_limit=2,
        save_steps=500,
        fp16=True,
        greater_is_better=True, 
        load_best_model_at_end=True,
        remove_unused_columns=False,
        metric_for_best_model="eval_f1",
        logging_dir="./logs_qa",
        logging_steps=50,
        report_to="none",
    )

    # Set up the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        compute_metrics=compute_metrics
    )
    # Start the fine-tuning
    trainer.train()

    # Save the fine-tuned model for future use
    model.save_pretrained("./fine_tuned_bert")
    tokenizer.save_pretrained("./fine_tuned_bert")

    # Print the evaluation results at the end of training
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

# Main execution function
def main():
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Prepare the data
    train_dataset, eval_dataset = prepare_data('csv/simple_squad.csv', 'csv/players_squad_format.csv')
    print("Training Sample:", train_dataset[:5])
    print("Evaluation Sample:", eval_dataset[:5])

    # Tokenize the data
    tokenized_train_dataset = tokenize_data(tokenizer, train_dataset)
    tokenized_eval_dataset = tokenize_data(tokenizer, eval_dataset)

    # Train the model
    train_model(model, tokenizer, tokenized_train_dataset, tokenized_eval_dataset)


# Execute the main function
if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.cuda.device_count() > 1:
            torch.distributed.destroy_process_group()
