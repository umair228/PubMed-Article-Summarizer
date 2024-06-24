from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


try:
    import sentencepiece
except ImportError as e:
    print("sentencepiece is not installed. Please install it using `pip install sentencepiece`.")
    raise e

# Step 1: Load the PubMed Summarization Dataset
dataset = load_dataset("ccdv/pubmed-summarization", "document")

# Step 2: Explore the Dataset
print(dataset)
print("Sample article: ", dataset['train'][0]['article'])
print("Sample abstract: ", dataset['train'][0]['abstract'])


# Step 3: Preprocess the Dataset
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples['article']]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['abstract'], max_length=150, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# Initialize the tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Apply preprocessing to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Step 4: Summarization Feature using T5
model = T5ForConditionalGeneration.from_pretrained(model_name)


def summarize_article(article):
    inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Example usage
sample_article = dataset['train'][0]['article']
summary = summarize_article(sample_article)
print("Summary: ", summary)

# Step 5: Allow user to input article
user_article = input("Please enter the article you want to summarize: ")
user_summary = summarize_article(user_article)
print("Summary: ", user_summary)
