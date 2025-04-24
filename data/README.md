# Preprocessing for Explainable Phishing Detection

This repository provides data and scripts for an explainable phishing detection project. It includes raw and preprocessed email datasets, as well as tools to generate custom train/test splits with balanced classes, plus two preprocessing pipelines (sterilization and BERT‑light).

## Directory Structure
```
.
├── data
│   ├── raw
│   ├── preprocessed
│   ├── train
│   └── test
├── create_train_test_sets.py
└── README.md
```

## Data

### Raw Data (`data/raw`)

Contains original downloaded datasets in their native formats (CSV files).

### Preprocessed Data (`data/preprocessed`)

Contains the combined and labeled dataset `all_datasets_combined.csv` with these columns:

- `text`: Original email content (classification feature)
- `p_label`: Phishing label (1 if phishing, 0 otherwise)
- `g_label`: Generated label (1 if machine-generated, 0 otherwise)
- `origin`: Origin dataset ID (see table below)

| Origin | Dataset Name                     | Type   | Source                                                                                                                               |
| ------ | -------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| 0      | phishing_dataset.csv             | phish  | Kaggle: https://www.kaggle.com/datasets/subhajournal/phishingemails                                                                 |
| 1      | jose private                     | phish  | Not used – unprocessable emails                                                                                                      |
| 3      | ceas-08.csv                      | phish  | Figshare: https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108                                               |
| 4      | enron.csv                        | phish  | Figshare: https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108                                               |
| 5      | ling.csv                         | phish  | Figshare: https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108                                               |
| 6      | assassin.csv                     | phish  | Figshare: https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108                                               |
| 7      | trec-05.csv                      | phish  | Figshare: https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108                                               |
| 8      | trec-06.csv                      | phish  | Figshare: https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108                                               |
| 9      | trec-07.csv                      | phish  | Figshare: https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108                                               |
| 10     | spam.csv                         | phish  | Kaggle: https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification                                 |
| 11     | combined_*_generated.csv         | mixed  | Kaggle: https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails                             |
| 12     | autextification_en_combined.csv  | gen    | Hugging Face: https://huggingface.co/datasets/symanto/autextification2023                                                              |
| 13     | ai_phishing_emails.csv           | mixed  | KSU: https://people.cs.ksu.edu/~lshamir/data/ai_phishing/                                                                             |

## Preprocessing and Dataset Splitting

The script `create_train_test_sets.py` loads `data/preprocessed/all_datasets_combined.csv`, performs optional filtering, balancing, two parallel text cleaning pipelines, and splits into train/test sets.

### Two text pipelines
- **Sterilized text** (`sterilized_text` column): full cleaning for classical models:
  1. Unicode normalize & HTML entity unescape → ASCII
  2. Lowercase
  3. Expand contractions
  4. Replace URLs → `URL`, emails → `EMAIL`
  5. Mask long digit runs (≥5 digits) → `<NUM_LONG>`, short runs (<5) → `<NUM>`
  6. Drop all non-alphanumeric characters (keep spaces)
  7. Collapse whitespace

- **BERT-light text** (`cleaned_text` column): minimal cleaning for BERT/DistilBERT:
  1. Unicode normalize & HTML entity unescape → ASCII
  2. Strip HTML tags (preserve visible text)

### Script Arguments

```bash
python3 create_train_test_sets.py [OPTIONS]
```

- `--output_name <name>` (str, **required**)  Base name for output files; creates CSVs in `data/train/train_<name>.csv` and `data/test/test_<name>.csv`.
- `--num_rows <int>`                       Limit total rows (after filtering). If omitted, selects largest balanced subset.
- `--origins <ints>`                       List of origin IDs to include (default: all).
- `--data_type <str>`                      One of `phishing`, `machine`, `mixed`, `any` (default: `any`).
- `--test_size <float>`                    Fraction for test split (default: 0.2).
- `--input_file <path>`                    Path to combined CSV (default: `data/preprocessed/all_datasets_combined.csv`).
- `--target_balance <float>`               Desired positive fraction (e.g., 0.5). If provided with `--balance_tolerance`, enforces class balance.
- `--balance_tolerance <float>`            Allowed deviation (e.g., 0.05 for ±5%).

### Balancing Behavior
- If `--target_balance` and `--balance_tolerance` are set:
  - With `--num_rows`: finds the largest feasible sample ≤ `num_rows` matching balance constraints.
  - Without `--num_rows`: samples a perfectly balanced subset (equal pos/neg) of maximum size.
- If no balance args but `--num_rows` provided: random sampling of that size.

### Filtering by Type
- `phishing`: filters to rows where `p_label` is non-null; balances on `p_label`.
- `machine`: filters to rows where `g_label` is non-null; balances on `g_label`.
- `mixed`: filters to rows where both `p_label` and `g_label` are non-null; balances on both (iterative search).
- `any`: filters to rows where `p_label` is non-null; balances on `p_label`.

### Example Usage

```bash
python3 create_train_test_sets.py \
  --output_name phishing_trial_a \
  --num_rows 5000 \
  --origins 0 3 4 \
  --data_type phishing \
  --test_size 0.25 \
  --input_file data/preprocessed/all_datasets_combined.csv \
  --target_balance 0.5 \
  --balance_tolerance 0.05
```


