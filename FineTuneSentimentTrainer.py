import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set environment and device for training
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cpu")
print("Using device:", device)

# Step 1: Define a simple labeled dataset for sentiment classification (1 = positive, 0 = negative)
data = {
    "text": [
        "I love this product, it's amazing!",
        "This was a great purchase, very happy.",
        "Works exactly as described, highly recommend.",
        "Terrible quality, broke after one use.",
        "Complete waste of money, don't buy this.",
        "Disappointed with this purchase, not worth it.",
        "Product is bad, awful, not worth it.",
        "Impressed with the product, worth it.",
        "The best product I've ever used!",
        "Absolutely terrible, do not buy!",
        "Okay product, not great but not bad.",
        "A complete scam, avoid!",
        "Very satisfied with my purchase.",
        "This is a total ripoff.",
        "I'm so glad I bought this!",
        "Waste of money.",
        "Would buy again!",
        "Never buying this brand again.",
        "Great value for money.",
        "Extremely disappointed."
    ],
    "label": [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Step 2: Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Step 4: Load a pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 5: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Step 6: Define training arguments for fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Step 7: Initialize Hugging Face Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Step 8: Fine-tune the model
trainer.train()

# Step 9: Save the trained model and tokenizer locally
model_path = "./fine_tuned_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Step 10: Define a prediction function to classify new text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if predictions == 1 else "Negative"

# Step 11: Run sentiment prediction on new examples
test_examples = [
    "I'm really impressed with the quality!",
    "This product is absolutely useless."
]

for example in test_examples:
    print(f"Text: {example}")
    print(f"Sentiment: {predict_sentiment(example)}")
    print()
