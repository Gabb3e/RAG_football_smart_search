from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
model.to(device)  # Move the model to the device (GPU if available)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Load the dataset from a CSV file
df = pd.read_csv('query_answer_data.csv')

# Split the dataset into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.2)  # 80% train, 20% eval
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenize the inputs and outputs
def tokenize_function(examples):
    inputs = examples['query']  # The queries (inputs)
    targets = examples['answer']  # The answers (outputs)
    
    # Tokenize inputs and targets (with truncation)
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize the targets using 'text_target'
    labels = tokenizer(text_target=targets, max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]

     # Convert all tensors to the same device
    return model_inputs

# Apply the tokenization to training and evaluation datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# Convert the datasets to PyTorch format
tokenized_train_dataset.set_format("torch")
tokenized_eval_dataset.set_format("torch")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    report_to="none"
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# Start the fine-tuning
trainer.train()

# Save the fine-tuned model for future use
model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")