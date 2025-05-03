from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re


api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


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





def api_call(PROMPT_SYSTEM, mail):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": mail},
        ],
        stream=False
    )

    print(f"Total tokens: {response.usage.total_tokens}")
    return response

def split_response(response):
    p_label = None
    reason_lst = []
    text = response.choices[0].message.content.strip()
    print("tttttttttttttext", text)
    # Split into non-empty lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Line 0: P_LABEL
    label_line = lines[0]
    m_label = re.search(r"P_LABEL:\s*<?\s*(\d)\s*>?", label_line)

    # Line 1: REASON
    reason_line = lines[1]
    m_reason = re.search(r"REASON:<?(.+)>?", reason_line)

    p_label = int(m_label.group(1))
    reason_str = m_reason.group(1)
    reason_lst = [r.strip() for r in reason_str.split(",")]
    return  p_label, reason_lst


def feed_mail_to_llm():
    path = Path("data/test/test_llm_trial.csv")
    df = pd.read_csv(path)
    pred_p_label = []
    reasons_p_label = []
    for idx, row in df.iterrows():
        mail = row["text"]
        response = api_call(PROMPT_SYSTEM, mail)
        p_label, reason_lst = split_response(response)
        print(f"p_label: {p_label}")
        print(f"real p_label: {row["p_label"]}")
        print(f"reason_lst: {reason_lst}")
        pred_p_label.append(p_label)
        reasons_p_label.append(reason_lst)
    df_pred_p = pd.DataFrame({'pred_p_label': pred_p_label, 'reasons_p_label': reasons_p_label})
    df = pd.concat([df, df_pred_p], axis=1)
    df.to_csv("data/test/res.csv", index=False)
    print(df[['pred_p_label', 'p_label']])



feed_mail_to_llm()