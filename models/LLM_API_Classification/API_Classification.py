from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm


api_key = os.getenv("DEEPSEEK_API_KEY")
client  = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


PROMPT_SYSTEM = ("""
    You are a cybersecurity assistant specialized in detecting phishing emails.  
    When I give you the full text of an email, do the following:  
    1. Decide if it’s phishing (1) or safe (0).  
    2. Output exactly two lines:  
       • P_LABEL:<1 or 0>  
       • REASON:<comma-separated list of short buzzwords that drove your decision, chosen from:  
         Urgent, Suspicious Link, Generic Greeting, Spoofed Domain, Mismatched URL, Threatening Language,  
         Request for Credentials, Unexpected Attachment, Poor Grammar, Unusual Sender, Hover Discrepancy,  
         Sense of Scarcity, Emotional Manipulation, Link Shortener, Domain Inconsistency, Too Good to Be True, 
         Suspicious Reply-To, Incorrect Branding, Excessive Punctuation, Invoice-Style, Calendar Invite, IP Mismatch  
       >  
    3. Do not add any other text.
        """)


def api_call(prompt_system: str, mail: str):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": mail},
        ],
        max_tokens=800,
        stream=False
    )
    return response


def split_response(response):
    text = response.choices[0].message.content.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    m1 = re.search(r"P_LABEL:\s*<?\s*(\d)\s*>?", lines[0])
    m2 = re.search(r"REASON:<?(.+)>?", lines[1])

    try:
        p_label = int(m1.group(1))
    except Exception:
        p_label = None
    try:
        reasons = [r.strip() for r in m2.group(1).split(",")]
    except Exception:
        reasons = None

    return p_label, reasons


def batch_api_call(prompt_system, mails):
    results = [None] * len(mails)
    with ThreadPoolExecutor(max_workers=len(mails)) as pool:
        future_to_idx = { pool.submit(api_call, prompt_system, mail): i
            for i, mail in enumerate(mails)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


def call_api_and_process_output():
    df = pd.read_csv(Path("data/test/small_test_set.csv"))
    backup_path = Path("models/LLM_API_Classification/llm_results/results_backup.csv")
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    n  = len(df)
    pred_labels    = [None] * n
    reason_labels  = [None] * n

    batch_size = 6
    for start in tqdm(range(0, n, batch_size)):
        end       = min(start + batch_size, n)
        mails     = df["text"].iloc[start:end].tolist()
        responses = batch_api_call(PROMPT_SYSTEM, mails)
        for offset, response in enumerate(responses):
            pl, rl = split_response(response)
            idx    = start + offset
            pred_labels[idx]   = pl
            reason_labels[idx] = rl

        df_partial = pd.DataFrame({"pred_p_label":    pred_labels, "reasons_p_label": reason_labels})
        df_out = pd.concat([df, df_partial], axis=1)
        df_out.to_csv(backup_path, index=False)

    df_pred = pd.DataFrame({"pred_p_label":   pred_labels, "reasons_p_label": reason_labels})
    df_out = pd.concat([df, df_pred], axis=1)
    df_out.to_csv("models/LLM_API_Classification/llm_results/results_final.csv", index=False)


if __name__ == "__main__":
    call_api_and_process_output()
