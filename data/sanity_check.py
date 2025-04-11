import pandas as pd
from pathlib import Path

def count_rows_in_preprocessed():
    preprocessed_dir = Path("data/preprocessed")
    csv_files = sorted(preprocessed_dir.glob("*.csv"))

    print(f"{'File':<45} | {'Rows':>7}")
    print("-" * 55)

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"{file.name:<45} | {len(df):>7}")
        except Exception as e:
            print(f"{file.name:<45} |   ERROR ({e.__class__.__name__})")

if __name__ == "__main__":
    count_rows_in_preprocessed()
