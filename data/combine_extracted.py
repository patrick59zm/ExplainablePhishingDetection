import pandas as pd
from pathlib import Path

def combine_all_preprocessed_datasets():
    input_dir = Path("data/extracted")
    output_path = Path("data/extracted_combined") / "all_datasets_combined.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"text", "p_label", "g_label", "origin"}
    dfs = []

    for file in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(file)
        if required_columns.issubset(df.columns):
            dfs.append(df)
            print(f"Loaded {file.name} ({len(df)} rows)")
        else:
            print(f"Skipping {file.name}: missing one or more required columns")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nCombined dataset saved to: {output_path.resolve()}")
    print(f"Total rows: {len(combined)}")

# Run it
combine_all_preprocessed_datasets()
