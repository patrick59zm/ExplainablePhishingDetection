import numpy as np
from transformers import pipeline, AutoTokenizer
import pandas as pd
import shap
import lime.lime_text


def bert_predict(email, explanaination_method=None):
    explanation = None
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    phishing_pipeline = pipeline(
        "text-classification",
        model=f"models/bert_checkpoints/checkpoint-10560",  
        tokenizer=tokenizer,
        top_k=None,     
        max_length=512,
        truncation=True
    )
    # Make prediction
    prediction = phishing_pipeline(email)
    print(prediction)
    # Extract the predicted class and score
    label = "phishing" if prediction[0][0]['label'] == "LABEL_1" else "legit"
    result = (label, prediction[0][0]['score'])
    
    if explanaination_method == "shap":
        masker = shap.maskers.Text(tokenizer=tokenizer)
        explainer = shap.Explainer(
            phishing_pipeline,  
            masker=masker
        )
        
    
        shap_values = explainer([email])
        explanation = shap_values
    

    return result, explanation