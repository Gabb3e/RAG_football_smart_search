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

# Load pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', ignore_mismatched_sizes=True)
model.lm_head = nn.Linear(model.config.d_model, model.config.vocab_size)

for param in model.parameters(): # Freeze all layers in the model except the output head (lm_head)
    param.requires_grad = False  # Freeze all parameters
for param in model.lm_head.parameters(): # Unfreeze only the last layer (output head) for fine-tuning
    param.requires_grad = True  # Unfreeze the lm_head (output head)
for param in model.model.decoder.layers[-6:].parameters(): # Unfreeze only the last few layers of the decoder for fine-tuning
    param.requires_grad = True  # Unfreeze the last X layers of the decoder

model.to(device)  # Move the model to the device (GPU if available)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Load the Simple SQuAD dataset
df_squad = pd.read_csv('csv/simple_squad.csv')
#print(df_squad.head())
print(df_squad.columns)
df_qa = pd.read_csv('csv/qa_data.csv')
#print(df_qa.head())
print(df_qa.columns)

df_combined = pd.concat([df_squad, df_qa], ignore_index=True)

# Split the dataset into training and evaluation sets
train_df, eval_df = train_test_split(df_combined, test_size=0.2)  # 80% train, 20% eval
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenize the inputs and outputs
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples['context'], 
        examples['question'], 
        max_length=512, 
        truncation=True, 
        padding="max_length"
    )
    
    # Tokenize 'answer' (targets) and ensure that it is in string format
    # Checking if 'answer' is a list or a single string
    if isinstance(examples['answer'], list):
        # If it’s a list, tokenize each element
        labels = tokenizer(
            text_target=[str(ans) for ans in examples['answer']], 
            max_length=150, 
            truncation=True, 
            padding="max_length"
        )
    else:
        # If it's a single string, tokenize it directly
        labels = tokenizer(
            text_target=str(examples['answer']), 
            max_length=150, 
            truncation=True, 
            padding="max_length"
        )
    
    # Add the labels (targets) to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Apply the tokenization to training and evaluation datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# Convert the datasets to PyTorch format
tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

# Function to post-process model outputs to calculate metrics
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

# Custom compute_metrics function
def compute_metrics(eval_pred):
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
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)

# Start the fine-tuning
trainer.train()

# Save the fine-tuned model for future use
model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")