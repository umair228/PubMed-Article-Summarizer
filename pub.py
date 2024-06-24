from datasets import load_dataset
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

# Step 1: Load the PubMed Summarization dataset
dataset = load_dataset("ccdv/pubmed-summarization", "document")

# Step 2: Explore and preprocess the dataset
def preprocess_function(examples):
    inputs = [doc for doc in examples['article']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['abstract'], max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Step 3: Fine-tune the T5 model
model = T5ForConditionalGeneration.from_pretrained(model_name)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

trainer.train()

# Step 4: Implement the summarization feature
def summarize_text(text, model, tokenizer):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
user_input = "Your article text here"
summary = summarize_text(user_input, model, tokenizer)
print("Summary:", summary)
