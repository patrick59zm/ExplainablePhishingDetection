import pandas as pd
from datasets import load_dataset, Value
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import os
import shap
import argparse
from transformers import pipeline
from metrics import moRF_LeRF_variable_length
from pathlib import Path
import lime.lime_text



def huggingface_predict_proba(texts):
    """
    Returns a 2D array of shape (len(texts), 2),
    where columns are probabilities for LABEL_0 and LABEL_1, respectively.
    """
    results = []
    for text in texts:
        output = phishing_pipeline(text)[0]  # ensure it returns all labels
        # Convert label-scores into a dictionary: e.g. {'LABEL_0': 0.0000048929, 'LABEL_1': 0.9999951124}
        label_dict = {item['label']: item['score'] for item in output}  # output[0] is the list of dicts
        # Ensure your label order is consistent: let's say [LABEL_0, LABEL_1]
        proba = [
            label_dict.get("LABEL_0", 0.0),
            label_dict.get("LABEL_1", 0.0)
        ]
        results.append(proba)
    return np.array(results)


# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--checkpoint_name', type=str, default='my-phishing-model/checkpoint-24747')
parser.add_argument('--samples_to_explain', type=int, default=100)
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--percent_dataset', type=int, default=100)



args = parser.parse_args()



device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU")

### Load the datasets ####
train_dataset = load_dataset('csv', data_files={
    'train': 'data/train/train_phishing_trial_a.csv'
}, split=f'train[:{args.percent_dataset}%]')
test_dataset = load_dataset('csv', data_files={
    'test': 'data/test/test_phishing_trial_a.csv'
}, split=f'test[:{args.percent_dataset}%]')

keep = {"p_label", "cleaned_text"}

train_dataset = train_dataset.select_columns(list(keep))
test_dataset = test_dataset.select_columns(list(keep))

train_dataset = train_dataset.filter(lambda x: x["cleaned_text"] is not None)
test_dataset = test_dataset.filter(lambda x: x["cleaned_text"] is not None)

train_dataset = train_dataset.cast_column("p_label", Value("int32")) 
test_dataset = test_dataset.cast_column("p_label", Value("int32"))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

def tokenize_function(examples):
    return tokenizer(examples["cleaned_text"], truncation=True, padding=True)

encoded_dataset_train = train_dataset.map(tokenize_function, batched=True)
encoded_dataset_test = test_dataset.map(tokenize_function, batched=True)



if args.train:
    # Rename columns so Trainer knows which are inputs vs. labels
    encoded_dataset_train = encoded_dataset_train.rename_column("p_label", "labels")
    encoded_dataset_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    encoded_dataset_test = encoded_dataset_test.rename_column("p_label", "labels")
    encoded_dataset_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(encoded_dataset_train)    # Training arguments
    training_args = TrainingArguments(
        output_dir="SHAP/checkpoints",
        evaluation_strategy="epoch",   # or "steps" if you have a validation set
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500
    )
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_test,  # if you have a validation set
    )

    trainer.train()

if args.test:
    # Set test dataset
    X_test = encoded_dataset_test["cleaned_text"][:args.samples_to_explain]
    y_test = encoded_dataset_test["p_label"][:args.samples_to_explain]


    # Create pipeline for SHAP (so that the features passed to SHAP are the words and not the tokens)
    phishing_pipeline = pipeline(
        "text-classification",
        model="SHAP/checkpoints/checkpoint-5439",  
        tokenizer=tokenizer,
        return_all_scores=True,     
        max_length=512,
        truncation=True
    )

    lime_explainer = lime.lime_text.LimeTextExplainer(
    class_names=["legit", "phishing"],
    )

    lime_val = lime_explainer.explain_instance(
    "Hi! this is a phishing email. Click the following link and we will steal your bank account data: phising_link.com",  # The text you want to explain
    huggingface_predict_proba,  # The model you want to explain
    num_features=1000,  # The number of features to show
    num_samples=1000,  # The number of samples to use for LIME
    )
    print(lime_val.as_list())
    


