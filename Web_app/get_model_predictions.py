import numpy as np
from transformers import pipeline, AutoTokenizer
import pandas as pd
import shap
import lime.lime_text
from models.bert import huggingface_predict_proba


def bert_predict(email, explanation_method=None):
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
    # Extract the predicted class and score
    label = "phishing" if prediction[0][0]['label'] == "LABEL_1" else "legit"
    result = (label, prediction[0][0]['score'])
    
    if explanation_method == "shap":
        masker = shap.maskers.Text(tokenizer=tokenizer)
        explainer = shap.Explainer(
            phishing_pipeline,  
            masker=masker
        )
        
    
        shap_values = explainer([email])
        explanation = shap_values
    elif explanation_method == "lime":
        lime_explainer = lime.lime_text.LimeTextExplainer(
        class_names=["legit", "phishing"],
        )
        explanation = lime_explainer.explain_instance(email,lambda x: huggingface_predict_proba(x, phishing_pipeline), num_features=1000, num_samples=5000)  


    return result, explanation