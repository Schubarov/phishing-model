import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load your dataset
file_path = "phish.csv"
df = pd.read_csv(file_path)

# 2. Preprocess the dataset
# Ensure the dataset has a column 'text' with your text data
df = df.rename(columns={"EmailText": "text"})
dataset = Dataset.from_pandas(df)

# 3. Tokenize the data
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Tokenize and add labels (identical to input_ids)
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()

# Apply tokenization with labels
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Load a pre-trained language model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")  # A smaller version of GPT-2

# 5. Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
)

# 6. Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 7. Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")