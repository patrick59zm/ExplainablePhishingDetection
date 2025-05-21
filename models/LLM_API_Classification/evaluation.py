import pandas as pd
import ast
import matplotlib.pyplot as plt
from pathlib import Path

def evaluation(file: Path, out_image: Path = Path("models/LLM_API_Classification/llm_results/phis_reasons_histogram.png")):
    # 1) Load
    df = pd.read_csv(file)

    # 2) Accuracy
    accuracy = (df['p_label'] == df['pred_p_label']).mean()
    print(f"Accuracy: {accuracy:.2%}")

    # 3) Safe parse
    def _safe_parse(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    df['reasons_list'] = df['reasons_p_label'].apply(_safe_parse)

    # 4) Flatten
    all_reasons = [r for sub in df['reasons_list'] for r in sub]

    # 5) Count
    reason_counts = pd.Series(all_reasons).value_counts()

    # 6) Plot & save
    plt.figure(figsize=(10,6))
    reason_counts.plot(kind='bar')
    plt.xlabel('Reason')
    plt.ylabel('Frequency')
    plt.title('Frequency of Predicted Reasons')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_image)
    plt.close()  # free memory

    print(f"Saved histogram to {out_image}")

def evaluation_m(file: Path, out_image: Path = Path("models/LLM_API_Classification/llm_results/m_reasons_histogram.png")):
    # 1) Load
    df = pd.read_csv(file)

    # 2) Accuracy
    accuracy = (df['g_label'] == df['pred_g_label']).mean()
    print(f"Accuracy: {accuracy:.2%}")
    print(df['g_label'] , df['pred_g_label'])
    # 3) Safe parse
    def _safe_parse(x):
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    df['reasons_list'] = df['reasons_g_label'].apply(_safe_parse)

    # 4) Flatten
    all_reasons = [r for sub in df['reasons_list'] for r in sub]

    # 5) Count
    reason_counts = pd.Series(all_reasons).value_counts()

    # 6) Plot & save
    plt.figure(figsize=(10,6))
    reason_counts.plot(kind='bar')
    plt.xlabel('Reason')
    plt.ylabel('Frequency')
    plt.title('Frequency of Predicted Reasons')
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