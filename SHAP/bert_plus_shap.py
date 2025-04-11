import pandas as pd
from datasets import load_dataset
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





# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--samples_to_explain', type=int, default=100)
parser.add_argument('--steps', type=int, default=5)



args = parser.parse_args()


# WARNING: Change the device to CUDA if running on NVIDIA GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


#Create csv files for train and test if they do not exist 
if not (os.path.exists('train.csv')) or (not os.path.exists('test.csv')):
    # Load your dataset
    df = pd.read_csv('your_dataset.csv')

    # Split the dataset into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the train and test sets to CSV files
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)

# Load the datasets
train_dataset = load_dataset('csv', data_files={
    'train': 'train.csv'
}, split='train')
test_dataset = load_dataset('csv', data_files={
    'test': 'test.csv'
}, split='test')

# Define Model and Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

def tokenize_function(examples):
    return tokenizer(examples["text_combined"], truncation=True, padding=True)

encoded_dataset_train = train_dataset.map(tokenize_function, batched=True)
encoded_dataset_test = test_dataset.map(tokenize_function, batched=True)



if args.train:
    # Rename columns so Trainer knows which are inputs vs. labels
    encoded_dataset_train = encoded_dataset_train.rename_column("label", "labels")
    encoded_dataset_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    encoded_dataset_test = encoded_dataset_test.rename_column("label", "labels")
    encoded_dataset_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="my-phishing-model",
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


# Set test dataset
X_test = encoded_dataset_test["text_combined"][:args.samples_to_explain]
y_test = encoded_dataset_test["label"][:args.samples_to_explain]


# Create pipeline for SHAP (so that the features passed to SHAP are the words and not the tokens)
phishing_pipeline = pipeline(
    "text-classification",
    model="my-phishing-model/checkpoint-24747",  
    tokenizer=tokenizer,
    return_all_scores=True,     
    max_length=512,
    truncation=True
)

masker = shap.maskers.Text(tokenizer=tokenizer)

explainer = shap.Explainer(
    phishing_pipeline,  
    masker=masker
)

shap_values = explainer(X_test)


### MoRF evaluation ###
print("MoRF evaluation: ")
fractions_removed_list, performance_list = moRF_LeRF_variable_length(
    phishing_pipeline,
    y_test,
    shap_values,
    metric=accuracy_score,
    steps=args.steps,
    mask_top=True
)
for i in range(len(fractions_removed_list)):
    print(f"Fraction removed: {fractions_removed_list[i]}, Performance: {performance_list[i]}")

### LeRF evaluation ###
print("LeRF evaluation: ")
fractions_removed_list, performance_list = moRF_LeRF_variable_length(
    phishing_pipeline,
    y_test,
    shap_values,
    metric=accuracy_score,
    steps=args.steps,
    mask_top=False
)
for i in range(len(fractions_removed_list)):
    print(f"Fraction removed: {fractions_removed_list[i]}, Performance: {performance_list[i]}")