import pandas as pd
from pathlib import Path


def convert_phishing_dataset():
    # Input and output paths
    input_path = Path("raw/Phishing_Email.csv")
    output_path = Path("preprocessed/phishing_dataset.csv")

    # Load raw data
    df = pd.read_csv(input_path)

    # Clean: drop rows with missing or empty email text
    df = df[df["Email Text"].notnull()]
    df = df[df["Email Text"].str.strip().astype(bool)]

    # Construct new DataFrame
    processed_df = pd.DataFrame({
        "text": df["Email Text"],
        "p_label": df["Email Type"].map({"Phishing Email": 1, "Safe Email": 0}),
        "g_label": None,  # Unknown if generated
        "origin": 0  # Dataset origin ID
    })
    print(processed_df.head())
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessed file
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Saved preprocessed dataset to: {output_path.resolve()}")


def convert_seven_phishing_datasets():
    input_dir = Path("raw")
    output_dir = Path("preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_files = [
        "Assassin.csv",
        "CEAS-08.csv",
        "Enron.csv",
        "Ling.csv",
        "TREC-05.csv",
        "TREC-06.csv",
        "TREC-07.csv"
    ]

    for origin_id, filename in enumerate(selected_files, start=3):
        file_path = input_dir / filename
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            continue
        if(filename == "Assassin.csv"):
            print("check 1")
        try:
            df = pd.read_csv(
                file_path,
                engine="python",
                quotechar='"',
                sep=",",
                encoding="utf-8",
                on_bad_lines="skip"
            )

        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")
            continue

        if not {"body", "label"}.issubset(df.columns):
            print(f"⚠️ Skipping {filename}: missing 'body' or 'label' columns")
            continue

        df = df[df["label"].astype(str).str.strip() != ""]
        df = df[pd.to_numeric(df["label"], errors="coerce").notnull()]
        df["p_label"] = df["label"].astype(int)


        df_clean = pd.DataFrame({
            "text": df["body"].astype(str).str.strip(),
            "p_label": df["p_label"],
            "g_label": None,
            "origin": origin_id
        })

        df_clean = df_clean[df_clean["text"] != ""]

        output_name = file_path.stem.lower() + ".csv"
        df_clean.to_csv(output_dir / output_name, index=False)

        # print(df_clean.head())
        # print(df_clean.columns)
        # print(f"✅ Processed {filename} → origin={origin_id} → saved to {output_name}")

def convert_spam_dataset():
    input_path = Path("raw/spam.csv")
    output_path = Path("preprocessed/spam.csv")
    origin_id = 10

    # Load the file with proper column names
    df = pd.read_csv(input_path, encoding="ISO-8859-1")


    # Rename and clean
    df = df.rename(columns={"v1": "label", "v2": "body"})

    # Filter out bad entries
    df = df[df["label"].notnull() & df["body"].notnull()]
    df = df[df["body"].astype(str).str.strip() != ""]

    # Map ham/spam to 0/1
    df["p_label"] = df["label"].map({"ham": 0, "spam": 1})

    # Final format
    df_final = pd.DataFrame({
        "text": df["body"].astype(str).str.strip(),
        "p_label": df["p_label"],
        "g_label": None,
        "origin": origin_id
    })

    # Save to preprocessed folder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"✅ Processed spam.csv → {output_path.resolve()}")

def convert_phishing_legit_generated():
    input_dir = Path("raw")
    output_dir = Path("preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("phishing_h.csv", 1, 0),
        ("phishing_m.csv", 1, 1),
        ("legit_h.csv", 0, 0),
        ("legit_m.csv", 0, 1)
    ]

    origin_id = 11
    combined_data = []

    for filename, p_label, g_label in files:
        file_path = input_dir / filename

        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            continue

        try:
            df = pd.read_csv(file_path, encoding="utf-8", engine="python", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1", engine="python", on_bad_lines="skip")

        if "text" not in df.columns:
            print(f"⚠️ Skipping {filename}: no 'text' column found")
            continue

        df = df[df["text"].notnull()]
        df = df[df["text"].astype(str).str.strip() != ""]

        df_clean = pd.DataFrame({
            "text": df["text"].astype(str).str.strip(),
            "p_label": p_label,
            "g_label": g_label,
            "origin": origin_id
        })

        combined_data.append(df_clean)
        print(f"✅ Processed {filename} ({len(df_clean)} entries)")

    if combined_data:
        merged_df = pd.concat(combined_data, ignore_index=True)
        output_path = output_dir / "combined_phishing_legit_generated.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Combined file saved to: {output_path.resolve()}")
    else:
        print("⚠️ No data processed. Combined file not created.")


def preprocess_autextification():
    input_dir = Path("raw")
    output_dir = Path("preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    origin_id = 12

    # Attribution with anonymized model labels A-F (these were misnamed as detection0)
    attribution_files = ["train_gen_detection0.tsv", "test_gen_detection0.tsv"]

    # Real detection files with "human" / "generated"
    detection_files = ["train_gen_detection1.tsv", "test_gen_detection1.tsv"]

    dfs = []

    # Attribution part (labels are model IDs: A–F → g_label = 1)
    for fname in attribution_files:
        path = input_dir / fname
        try:
            df = pd.read_csv(path, sep="\t", usecols=["text", "label"], engine="python")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep="\t", encoding="ISO-8859-1", usecols=["text", "label"], engine="python")

        df = df[df["text"].notnull() & df["label"].notnull()]
        df = df[df["text"].astype(str).str.strip() != ""]

        df_clean = pd.DataFrame({
            "text": df["text"].astype(str).str.strip(),
            "p_label": None,
            "g_label": 1,  # All are generated
            "origin": origin_id
        })

        dfs.append(df_clean)
        print(f"✅ Processed attribution file (was detection0): {fname} ({len(df_clean)} rows)")

    # Detection part (labels are 'human' or 'generated')
    for fname in detection_files:
        path = input_dir / fname
        try:
            df = pd.read_csv(path, sep="\t", usecols=["text", "label"], engine="python")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep="\t", encoding="ISO-8859-1", usecols=["text", "label"], engine="python")

        df = df[df["text"].notnull() & df["label"].notnull()]
        df = df[df["text"].astype(str).str.strip() != ""]

        df_clean = pd.DataFrame({
            "text": df["text"].astype(str).str.strip(),
            "p_label": None,
            "g_label": df["label"].map({"human": 0, "generated": 1}),
            "origin": origin_id
        })

        dfs.append(df_clean)
        print(f"✅ Processed true detection file: {fname} ({len(df_clean)} rows)")

    # Combine and save
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        output_path = output_dir / "autextification_en_combined.csv"
        combined.to_csv(output_path, index=False)
        print(f"✅ Saved combined dataset to: {output_path.resolve()}")
    else:
        print("⚠️ No data processed.")


def preprocess_ai_phishing_emails():
    input_dir = Path("raw/AI_phishing_emails")
    output_dir = Path("preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    origin_id = 13
    emails = []

    for file in sorted(input_dir.glob("email_*.txt")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    emails.append({
                        "text": content,
                        "p_label": 1,
                        "g_label": 1,
                        "origin": origin_id
                    })
        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")

    if emails:
        df = pd.DataFrame(emails)
        output_path = output_dir / "ai_phishing_emails.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} AI phishing emails to: {output_path.resolve()}")
    else:
        print("⚠️ No valid emails found.")

# Run it



convert_phishing_dataset()
convert_seven_phishing_datasets()
convert_spam_dataset()
convert_phishing_legit_generated()
preprocess_autextification()
preprocess_ai_phishing_emails()