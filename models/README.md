# Models and Explainability

This directory contains all trained classification models and their associated explainability setups for phishing detection and machine-generated email recognition.

## Folder Structure

```
models/
├── LLM_API_Classification/           # DeepSeek API wrapper and example outputs
│   ├── llm_results/                  # Sample DeepSeek API responses (JSON/text)
│   ├── __init__.py
│   ├── API_Classification.py         # Wrapper for phishing classification via DeepSeek
│   ├── API_Classification_gen.py     # Wrapper for machine-generated-mail detection via DeepSeek
│   └── evaluation.py                 # Evaluation scripts for LLM-based predictions
├── __init__.py
├── bert.py                           # BERT model loading, inference, and embedding extraction
├── logreg_explainability.py          # Logistic Regression + ELI5 explanations
├── train_logreg.py                   # Train & serialize logistic regression model
├── train_xgboost.py                  # Train & serialize XGBoost model
├── xgboost_explainability.py         # XGBoost + SHAP-based explanations
├── logistic_regression_model_pipeline.joblib
├── xgboost_model_pipeline.joblib
metrics.py                        # Aggregation & plotting of all model metrics
```

## Supported Models

1. **Logistic Regression**

   * Trained on sterilized text features.
   * Explanations via ELI5-based feature weights.
2. **XGBoost**

   * Tree-based classifier on sterilized text features.
   * Explanations via SHAP values (global & local insights).
3. **BERT-based Classifiers**

   * Fine-tuned BERT encoder for phishing vs. safe and for machine-generated vs. human-written.
   * Two XAI strategies:

     * **LIME** (Local interpretable perturbations)
     * **SHAP** (Transformer-specific token-attribution)
4. **DeepSeek LLM**

   * `deepseek-chat` endpoint for both classification and natural-language explanations.
   * Handles phishing detection and machine-generated email recognition in one API call.

## Training and Serialization

* **`train_logreg.py`** and **`train_xgboost.py`** load the preprocessed datasets (`data/train/*.csv`), train the models, evaluate on held-out test sets, and serialize pipelines to `.joblib` files.
* **`bert.py`** includes functions to load a pretrained/fine-tuned BERT encoder, run inference, and extract embeddings for XAI.

## Explainability Pipelines

* **`logreg_explainability.py`**: Generates per-feature weight explanations (top predictors).
* **`xgboost_explainability.py`**: Wraps SHAP TreeExplainer to produce local and global feature importance.
* **`API_Classification.py`** & **`API_Classification_gen.py`**: Interact with DeepSeek for natural-language explanations and verdicts.

## Evaluation

* **`LLM_API_Classification/evaluation.py`** runs LLM-based predictions against ground truth.
* **`metrics.py`** aggregates results and can produce summary tables or plots comparing all models.

---

## Usage Example

```bash
# Train and serialize local models
python3 models/train_logreg.py --config configs/logreg.yaml
python3 models/train_xgboost.py --config configs/xgb.yaml

# Run explanations on a sample email
python3 models/logreg_explainability.py --input "sample_email.txt"
python3 models/xgboost_explainability.py --input "sample_email.txt"

# Use DeepSeek for classification & explanation
python3 -c "from models.LLM_API_Classification.API_Classification import classify; print(classify('sample_email.txt'))"
```

## Notes

* Ensure preprocessed datasets are available under `data/` before training.
* Set `DEEPSEEK_API_KEY` in your environment to use LLM classification:

  ```bash
  export DEEPSEEK_API_KEY="sk-..."
  ```

---

*This README serves as a quick reference for navigating and using the `models` submodule in the Explainable Phishing Detection project.*
