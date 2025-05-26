import pandas as pd
import ast
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

def evaluation(file: Path,
               out_image: Path = Path("models/LLM_API_Classification/llm_results/phis_reasons_histogram.png")):
    # 1) Load
    df = pd.read_csv(file)
    y_true = df['p_label']
    y_pred = df['pred_p_label']

    # 2) Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

    # 3) Recall (macro)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Recall (macro): {recall:.2%}")

    # 4) F1 Score (macro)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"F1 Score (macro): {f1:.2%}")

    # 5) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # (Optional) Save a histogram or any other plot to out_image...

    fig, ax = plt.subplots()
    df['pred_p_label'].hist(ax=ax)
    fig.savefig("models/LLM_API_Classification/llm_results/phis_hist.png")

    # 3) Safe parse reasons into lists
    def _safe_parse(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    df['reasons_list'] = df['reasons_p_label'].apply(_safe_parse)

    # 4) Keep only rows where the true label is positive
    positive_df = df[df['p_label'] == 1]

    # 5) Flatten just those positive‐label reasons
    all_reasons = [reason.lower().strip()
                   for sublist in positive_df['reasons_list']
                   for reason in sublist]

    # 6) Count and plot
    reason_counts = pd.Series(all_reasons).value_counts()
    reason_counts = reason_counts[reason_counts > 37]

    plt.figure(figsize=(5, 4))
    reason_counts.plot(kind='bar')
    plt.xlabel('Top-10 Reasons')
    plt.ylabel('Frequency')
    plt.title('Frequency of Reasons (positive predictions)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_image)
    plt.close()

    print(f"Saved histogram to {out_image}")
def evaluation_m(file: Path, out_image: Path = Path("models/LLM_API_Classification/llm_results/m_reasons_histogram.png")):
    # 1) Load
    df = pd.read_csv(file)
    y_true = df['g_label']
    y_pred = df['pred_g_label']

    # 2) Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

    # 3) Recall (macro)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Recall (macro): {recall:.2%}")

    # 4) F1 Score (macro)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"F1 Score (macro): {f1:.2%}")

    # 5) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # (Optional) Save a histogram or any other plot to out_image...

    fig, ax = plt.subplots()
    df['pred_g_label'].hist(ax=ax)
    fig.savefig("models/LLM_API_Classification/llm_results/phis_hist.png")

    # 3) Safe parse
    def _safe_parse(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    df['reasons_list'] = df['reasons_g_label'].apply(_safe_parse)
    positive_df = df[df['g_label'] == 1]
    # 4) Flatten
    all_reasons = [reason.lower().strip()
                   for sublist in positive_df['reasons_list']
                   for reason in sublist]

    # 5) Count
    reason_counts = pd.Series(all_reasons).value_counts()
    reason_counts = reason_counts[reason_counts > 5]

    # 6) Plot & save
    plt.figure(figsize=(10,6))
    reason_counts.plot(kind='bar')
    plt.xlabel('Reason')
    plt.ylabel('Frequency')
    plt.title('Frequency of Reasons (True Positives)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_image)
    plt.close()  # free memory

    print(f"Saved histogram to {out_image}")


# Example usage:
in_path = Path("models/LLM_API_Classification") / "llm_results" / "results_final.csv"
evaluation(in_path)

in_p2 = Path("models/LLM_API_Classification") / "llm_results" / f"results_final_m.csv"
evaluation_m(in_p2)