from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd


# Load pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

df = pd.read_csv('/home/g2de/documents/programmering/projects/transfer_hub/query_answer_data.csv')
dataset = Dataset.from_pandas(df)

# Tokenize the inputs and outputs
def tokenize_function(examples):
    inputs = examples['query']  # The queries (inputs)
    targets = examples['answer']  # The answers (outputs)
    
    # Tokenize inputs and targets (with truncation)
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Tokenize the targets (answers)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the tokenization
tokenized_dataset = dataset.map(tokenize_function)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start the fine-tuning
trainer.train()

# Save the fine-tuned model for future use
model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")