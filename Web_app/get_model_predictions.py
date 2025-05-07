import numpy as np
from transformers import pipeline, AutoTokenizer
import pandas as pd


def bert_predict(email):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    phishing_pipeline = pipeline(
        "text-classification",
        model=f"models/bert_checkpoints/checkpoint-3000",  
        tokenizer=tokenizer,
        top_k=None,     
        max_length=512,
        truncation=True
    )
    # Make prediction
    prediction = phishing_pipeline(email)
    # Extract the predicted class and score
    if prediction[0][0]['score'] > 0.5:
        return "legit", prediction[0][0]['score']
    else:
        return "phishing", prediction[0][1]['score']