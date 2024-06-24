import requests
from flask import Flask, render_template, url_for
from flask import request as req
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = Flask(__name__)

# Load the PubMed Summarization Dataset and the T5 model
model_name = 't5-small'
dataset = load_dataset("ccdv/pubmed-summarization", "document")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_article(article):
    inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route("/", methods=["GET", "POST"])
def Index():
    return render_template("index.html")

@app.route("/Summarize", methods=["GET", "POST"])
def Summarize():
    if req.method == "POST":
        data = req.form["data"]
        summary_method = req.form["summary_method"]

        if summary_method == "huggingface":
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": "Bearer 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"}
            maxL = int(req.form["maxL"])
            minL = maxL // 4

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            output = query({"inputs": data, "parameters": {"min_length": minL, "max_length": maxL}})[0]
            summary = output["summary_text"]
        else:
            summary = summarize_article(data)

        return render_template("index.html", result=summary)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

