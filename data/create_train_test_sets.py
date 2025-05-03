import pandas as pd
from pathlib import Path
import math
from sklearn.model_selection import train_test_split
import re
import unicodedata
from contractions import fix as expand_contractions
from bs4 import BeautifulSoup
import html
import argparse

def balance_sample(df, label, requested_num_rows, target_balance, balance_tolerance):
    """
    For single-label data, determine the maximum sample size <= requested_num_rows
    that can achieve a balance (for the given label) within target_balance ± balance_tolerance.
    Then, sample that many rows.

    Here, a feasible sample of size S is one where there exists an integer P (number of positives)
    in [ceil((target_balance - tol)*S), floor((target_balance + tol)*S)]
    such that P <= available positives and (S - P) <= available negatives.
    """
    positives = df[df[label] == 1]
    negatives = df[df[label] == 0]
    N_pos = len(positives)
    N_neg = len(negatives)

    feasible_S = None
    chosen_P = None
    # Try S from the requested number down to 1.
    for S in range(requested_num_rows, 0, -1):
        P_min = math.ceil((target_balance - balance_tolerance) * S)
        P_max = math.floor((target_balance + balance_tolerance) * S)
        # Compute target candidate as round(target_balance * S), then clamp to [P_min, P_max]
        candidate = round(target_balance * S)
        candidate = max(P_min, min(candidate, P_max))
        if candidate <= N_pos and (S - candidate) <= N_neg:
            feasible_S = S
            chosen_P = candidate
            break
    if feasible_S is None:
        raise ValueError("Could not find any feasible sample size under the given constraints.")
    if feasible_S < requested_num_rows:
        print(
            f"Requested sample size {requested_num_rows} not feasible. Using maximum possible size {feasible_S} instead.")
    pos_sample = positives.sample(n=chosen_P, random_state=42)
    neg_sample = negatives.sample(n=feasible_S - chosen_P, random_state=42)
    return pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)


def balance_sample_mixed(df, requested_num_rows, target_balance, balance_tolerance, max_attempts=1000):
    """
    For mixed data, iterate from the requested sample size S downwards,
    and for each S attempt max_attempts to find a random subset where both
    p_label and g_label have positive ratios within [target_balance - tol, target_balance + tol].
    """
    for S in range(requested_num_rows, 0, -1):
        for attempt in range(max_attempts):
            subset = df.sample(n=S, random_state=42 + attempt)
            ratio_p = (subset["p_label"] == 1).mean()
            ratio_g = (subset["g_label"] == 1).mean()
            if ((target_balance - balance_tolerance) <= ratio_p <= (target_balance + balance_tolerance) and
                    (target_balance - balance_tolerance) <= ratio_g <= (target_balance + balance_tolerance)):
                if S < requested_num_rows:
                    print(
                        f"Requested sample size {requested_num_rows} not feasible. Using maximum possible size {S} instead.")
                return subset
    raise ValueError("Could not find any feasible balanced subset for mixed data under the given constraints.")


def print_balance_info(df, data_type):
    """
    Print the percentage of positive and negative samples in the DataFrame.
    For 'phishing', 'machine', and 'any', the balance is computed on one label;
    for 'mixed', both are reported.
    """
    if data_type in ["phishing", "any", "mixed"]:
        pos_rate = (df["p_label"] == 1).mean() * 100
        neg_rate = (df["p_label"] == 0).mean() * 100
        print(f"P_label: {pos_rate:.1f}% positive | {neg_rate:.1f}% negative")
    elif data_type == "machine":
        pos_rate = (df["g_label"] == 1).mean() * 100
        neg_rate = (df["g_label"] == 0).mean() * 100
        print(f"G_label: {pos_rate:.1f}% positive | {neg_rate:.1f}% negative")


def sterilize_phishing_text(s: str) -> str:
    """
    Full sterilization pipeline:
      1. Unicode normalize & unescape HTML entities → ASCII
      2. Lowercase
      3. Expand contractions (“can’t” → “cannot”)
      4. Replace URLs → URL, emails → EMAIL
      5. Mask long digit runs (5 or more digits) → <NUM_LONG>, short runs (digits < 5) → <NUM>
      6. Remove any remaining non-alphanumeric (keeps spaces)
      7. Collapse whitespace
    """
    s = html.unescape(s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    s = expand_contractions(s)
    s = re.sub(r'https?://\S+|www\.\S+', 'URL', s)
    s = re.sub(r'\S+@\S+', 'EMAIL', s)
    s = re.sub(r'\d{5,}', '<NUM_LONG>', s)
    s = re.sub(r'\b\d{1,4}\b', '<NUM>', s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_for_bert(s: str) -> str:
    """
       Minimal cleaning for BERT:
         1. Unicode normalize & unescape HTML entities → ASCII
         2. Strip HTML tags (keep visible text)
    """
    # 1) Unicode normalize & unescape
    s = html.unescape(s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    # 2) Strip HTML
    s = BeautifulSoup(s, "html.parser").get_text(separator=" ")
    return s

def filter_and_split_dataset(
        output_name: str,
        num_rows: int = None,
        origins: list = None,  # consult read_me for details
        data_type: str = "any",  # must be one of: "phishing", "machine", "mixed", "any"
        test_size: float = 0.2,
        input_file: str = "data/extracted_combined/all_datasets_combined.csv",
        target_balance: float = None,  # desired positive fraction (e.g., 0.5)
        balance_tolerance: float = None  # allowed deviation (e.g., 0.05 for ±5%)
     ):
    """
    This function filters the combined dataset and splits it into train/test sets.

    Data-type filtering:
      - 'phishing': only rows with non-null p_label are kept; balance is enforced on p_label.
      - 'machine': only rows with non-null g_label are kept; balance is enforced on g_label.
      - 'mixed': only rows with non-null p_label and g_label are kept; balance is enforced on both.
      - 'any': only rows with non-null p_label are required; balance is enforced on p_label.

    If num_rows is not specified, the function retrieves the largest balanced subset.
    Otherwise, if the desired configuration is not possible for the requested number of rows,
    it finds the biggest possible number that satisfies the constraints.

    After splitting into train and test, the function prints the % of positive and negative samples.
    """
    input_path = Path(input_file)
    print(input_path)
    df = pd.read_csv(input_path)

    # Filter by origins if provided.
    if origins is not None:
        df = df[df["origin"].isin(origins)]

    # Filter by data_type:
    if data_type == "phishing":
        df = df[df["p_label"].notna()]
        label_to_balance = "p_label"
    elif data_type == "machine":
        df = df[df["g_label"].notna()]
        label_to_balance = "g_label"
    elif data_type == "mixed":
        df = df[df["p_label"].notna() & df["g_label"].notna()]
        label_to_balance = "p_label"
    elif data_type == "any":
        df = df[df["p_label"].notna()]
        label_to_balance = "p_label"
    else:
        raise ValueError("Invalid data_type. Must be one of: 'phishing', 'machine', 'mixed', 'any'.")

    # Enforce balance if the parameters are provided.
    if target_balance is not None and balance_tolerance is not None:
        if num_rows is not None:
            # When a specific number of rows is desired:
            df = balance_sample(df, label_to_balance, num_rows, target_balance, balance_tolerance)
        else:
            # Without a specified number of rows, retrieve the largest balanced subset.
            if data_type in ["phishing", "any", "mixed"]:
                label_to_balance = "p_label"
            else:
                label_to_balance = "g_label"
            positives = df[df[label_to_balance] == 1]
            negatives = df[df[label_to_balance] == 0]
            # The largest balanced subset is twice the size of the smaller class.
            min_count = min(len(positives), len(negatives))
            pos_sample = positives.sample(n=min_count, random_state=42)
            neg_sample = negatives.sample(n=min_count, random_state=42)
            df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)
    else:
        # If no balance constraint is provided but num_rows is specified, sample randomly.
        if num_rows is not None:
            num_rows = min(num_rows, len(df))
            df = df.sample(n=num_rows, random_state=42)

    #Disregard rows with emtpy Text
    df = df.loc[lambda d: d["text"].notna() & d["text"].astype(str).str.strip().astype(bool)]

    #Cast labels to bool
    df['p_label'] = df['p_label'].apply(lambda v: None if pd.isna(v) else bool(v))
    df['g_label'] = df['g_label'].apply(lambda v: None if pd.isna(v) else bool(v))

    #Create the cleaned text
    df['sterilized_text'] = df['text'].astype(str).apply(sterilize_phishing_text)
    df['cleaned_text'] = df['text'].astype(str).apply(clean_for_bert)
    # Split into train/test sets.
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Define paths
    train_output = Path("data") / "train" / f"train_{output_name}.csv"
    test_output = Path("data") / "test" / f"test_{output_name}.csv"

    #Create directories if not already existent
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    #Saved train and test set
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"Train saved to: {train_output} ({len(train_df)} rows)")
    print(f"Test saved to: {test_output} ({len(test_df)} rows)")

    # Evaluate and print the balance in each set.
    print("Train set balance:")
    print_balance_info(train_df, data_type)
    print("Test set balance:")
    print_balance_info(test_df, data_type)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and split a dataset.")
    parser.add_argument("--output_name", type=str, required=True, help="Name used for the output files")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to sample")
    parser.add_argument("--origins", type=int, nargs="*", default=None, help="List of origin IDs to include")
    parser.add_argument("--data_type", type=str, default="any", choices=["phishing", "machine", "mixed", "any"],
                        help="Type of data to filter")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use as test set")
    parser.add_argument("--input_file", type=str, default="data/extracted_combined/all_datasets_combined.csv",
                        help="Input CSV file")
    parser.add_argument("--target_balance", type=float, default=0.5,
                        help="Desired fraction of positives (e.g., 0.5 for 50%)")
    parser.add_argument("--balance_tolerance", type=float, default=0.05,
                        help="Allowed deviation from target balance (e.g., 0.05)")

    args = parser.parse_args()

    filter_and_split_dataset(
        output_name=args.output_name,
        num_rows=args.num_rows,
        origins=args.origins,
        data_type=args.data_type,
        test_size=args.test_size,
        input_file=args.input_file,
        target_balance=args.target_balance,
        balance_tolerance=args.balance_tolerance,
    )
