



# Explainable Phishing Detection

### This is the repo of project 11 of the ETH Data Science Lab spring 2025

A modular project for detecting phishing and machine-generated emails using classical ML, transformer-based models, and an LLM-based API, with integrated explainability. You can preprocess data, train and evaluate multiple models, generate feature-based or natural-language explanations, and serve everything through a web interface.

## Repository Structure

```
ExplainablePhishingDetection/      # Root folder
├── data/                          # Raw and preprocessed datasets + splitting scripts
│   ├── raw/
│   ├── extracted/
│   ├── extracted_combined/
│   ├── train/
│   ├── test/
│   └── create_train_test_sets.py  # Custom splitting & preprocessing pipelines
├── models/                       # Trained models & XAI pipelines
│   ├── LLM_API_Classification/   # DeepSeek API wrappers & results
│   ├── bert.py
│   ├── train_logreg.py
│   ├── train_xgboost.py
│   ├── logreg_explainability.py
│   ├── xgboost_explainability.py
│   ├── logistic_regression_model_pipeline.joblib
│   ├── xgboost_model_pipeline.joblib
├── Web_app/                      # Front-end + back-end for interactive demo
│   ├── chat_bot.py
│   ├── front_end_setup.py
│   ├── get_model_predictions.py
│   ├── get_llm_prediction.py
│   ├── explanation_processing.py
├── bert_main.py                  # Standalone script for BERT-based batch inference
├── metrics.py                    # Global evaluation and plotting utilities
├── models_report_and_statistics.docx  # Detailed report of experiments
├── requirements.txt              # Core Python dependencies
├── LICENSE                       # Project license
└── README.md                     # (This file)
```

## Setup and Installation

1. Clone and enter the repository:

   ```bash
   git clone <https://github.com/patrick59zm/ExplainablePhishingDetection>
   cd ExplainablePhishingDetection
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install core dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Set your DeepSeek API key for LLM classification and explanation:

   ```bash
   export DEEPSEEK_API_KEY="sk-<your-key>"
   ```

## Data

The preprocessing scripts and raw data are provided under the `data/` directory. If you prefer not to generate your own splits, you can download preprocessed train and test sets from the following link:

```
https://polybox.ethz.ch/index.php/s/oS8Ej54GpfpLsP6
```

See `data/README.md` for details on data sources, pipelines, and how to run `create_train_test_sets.py`.

## Models

This project includes multiple classification models and explainability pipelines:

* Logistic Regression and XGBoost (classical ML)
* BERT-based classifiers with LIME and SHAP explanations
* DeepSeek LLM API for natural-language classification and explanation

If you do not want to train these models yourself, you can download the pretrained pipelines (`.joblib` files) here:

```
link_2
```

See `models/README.md` for instructions on training, serialization, and running explainability scripts.

## Web App Demo

The `Web_app/` directory contains a web application for interactive classification and explanation:

1. Build front-end assets:

   ```bash
   python3 -m Web_app.front_end_setup
   ```
   
2. Open your browser at the port printed in the console (default: `http://localhost:7860`).

Refer to `Web_app/README.md` for full instructions on using the demo.

## Evaluation and Reporting

Use `metrics.py` (root) and `models/metrics.py` to compute and visualize accuracy, precision, recall, F1, and ROC-AUC for all models. A detailed experimental report is available in `models_report_and_statistics.docx`.

## Usage Examples

```bash
# Generate train/test splits
python3 data/create_train_test_sets.py --output_name phishing_test --num_rows 10000 --data_type phishing

# Train classical models
python3 models/train_logreg.py
python3 models/train_xgboost.py

# Run explanations on a sample email
python3 models/logreg_explainability.py --input sample_email.txt

# Launch interactive web demo
python3 -m Web_app.front_end_setup
```

## Contributing

Contributions are welcome. Please fork the repository, implement your changes, and submit a pull request. Ensure new code is tested and documentation is updated.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.



