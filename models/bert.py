import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from metrics import moRF_LeRF_SHAP, moRF_LeRF_LIME
from sklearn.metrics import accuracy_score
import shap
import lime.lime_text
import pandas as pd
from sklearn.metrics import classification_report



def huggingface_predict_proba(texts, phishing_pipeline):
    """
    Returns a 2D array of shape (len(texts), 2),
    where columns are probabilities for LABEL_0 and LABEL_1, respectively.
    """
    results = []
    for text in texts:
        output = phishing_pipeline([text])[0]  # wrap text in a list to ensure compatibility
        # Convert label-scores into a dictionary: e.g. {'LABEL_0': 0.0000048929, 'LABEL_1': 0.9999951124}
        label_dict = {item['label']: item['score'] for item in output}  # output[0] is the list of dicts
        # Ensure your label order is consistent: let's say [LABEL_0, LABEL_1]
        proba = [
            label_dict.get("LABEL_0", 0.0),
            label_dict.get("LABEL_1", 0.0)
        ]
        results.append(proba)
    return np.array(results)

def train_bert(model, encoded_dataset_train, encoded_dataset_test, epochs=10):
    # Rename columns so Trainer knows which are inputs vs. labels
    encoded_dataset_train = encoded_dataset_train.rename_column("p_label", "labels")
    encoded_dataset_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    encoded_dataset_test = encoded_dataset_test.rename_column("p_label", "labels")
    encoded_dataset_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(encoded_dataset_train)    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/bert_checkpoints",
        eval_strategy="epoch", 
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=1000,
        
    )
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_test, 
    )

    trainer.train()

def test_bert(encoded_dataset_test, tokenizer, shap_explain, lime_explain, samples_to_explain=100, steps=5, checkpoint_name="", additional_metrics=False):
    # Evaluate the model
    # Set test dataset
    X_test = encoded_dataset_test["cleaned_text"]
    y_test = encoded_dataset_test["p_label"]


    # Create pipeline for SHAP (so that the features passed to SHAP are the words and not the tokens)
    phishing_pipeline = pipeline(
        "text-classification",
        model=f"models/bert_checkpoints/{checkpoint_name}",  
        tokenizer=tokenizer,
        top_k=None,     
        max_length=512,
        truncation=True
    )
    if shap_explain:
        masker = shap.maskers.Text(tokenizer=tokenizer)

        explainer = shap.Explainer(
            phishing_pipeline,  
            masker=masker
        )

        shap_values = explainer(X_test[:samples_to_explain])


        ### MoRF evaluation ###
        print("MoRF evaluation: ")
        fractions_removed_list, MoRF_performance_list = moRF_LeRF_SHAP(
            phishing_pipeline,
            y_test,
            shap_values,
            metric=accuracy_score,
            steps=steps,
            mask_top=True
        )
        for i in range(len(fractions_removed_list)):
            print(f"Fraction removed: {fractions_removed_list[i]}, MoRF Performance: {MoRF_performance_list[i]}")

        ### LeRF evaluation ###
        print("LeRF evaluation: ")
        _, LeRF_performance_list = moRF_LeRF_SHAP(
            phishing_pipeline,
            y_test,
            shap_values,
            metric=accuracy_score,
            steps=steps,
            mask_top=False
        )
        for i in range(len(fractions_removed_list)):
            print(f"Fraction removed: {fractions_removed_list[i]}, LeRF Performance: {LeRF_performance_list[i]}")
        
        results_df = pd.DataFrame({
            'Fraction Removed': fractions_removed_list,
            'MoRF_Performance': MoRF_performance_list,
            'LeRF_Performance': LeRF_performance_list
        })
        results_df.to_csv('results/SHAP_MoRF_LeRF_results.csv', index=False)
    if lime_explain:
        lime_explainer = lime.lime_text.LimeTextExplainer(
        class_names=["legit", "phishing"],
        )

        lime_val_list = []
        for i in range(samples_to_explain):
            lime_val = lime_explainer.explain_instance(X_test[i],lambda x: huggingface_predict_proba(x, phishing_pipeline), num_features=1000, num_samples=5000)  
            lime_val_list.append(lime_val)
        
        # MoRF Evaluation
        print("MoRF Evaluation")
        fractions_removed, MoRF_performances = moRF_LeRF_LIME(phishing_pipeline, X_test[:samples_to_explain], y_test, lime_val_list, metric=accuracy_score, steps=steps, MoRF=True)
        
        for i in range(len(fractions_removed)):
            print(f"Fraction removed: {fractions_removed[i]}, MoRF Performance: {MoRF_performances[i]}")
            
        # LeRF Evaluation
        print("LeRF Evaluation")
        _ , LeRF_performances = moRF_LeRF_LIME(phishing_pipeline, X_test[:samples_to_explain], y_test, lime_val_list, metric=accuracy_score, steps=steps, MoRF=False)
        for i in range(len(fractions_removed)):
            print(f"Fraction removed: {fractions_removed[i]}, LeRF Performance: {LeRF_performances[i]}")
        
        # Save the results to a CSV file
        results_df = pd.DataFrame({
            'Fraction Removed': fractions_removed,
            'MoRF_Performance': MoRF_performances,
            'LeRF_Performance': LeRF_performances
        })
        results_df.to_csv('results/LIME_MoRF_LeRF_results.csv', index=False)
    if additional_metrics:
        print("Additional metrics evaluation")
        # Add any additional metrics you want to compute here
        y_pred = []
        for i in range(len(X_test)):
            pred = phishing_pipeline(X_test[i])[0]
            neg_score = pred[0]["score"] 
            pos_score = pred[1]["score"]
            #print(f"neg_score: {neg_score}, pos_score: {pos_score}")
            if neg_score > pos_score:
                y_pred.append(0)
            else:
                y_pred.append(1)
        print(classification_report(y_test, y_pred))


