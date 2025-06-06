from openai import OpenAI
import os
import re


api_key = os.getenv("DEEPSEEK_API_KEY")
client  = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


PROMPT_SYSTEM_PHISHING = ("""
    You are a cybersecurity assistant specialized in detecting phishing emails.  
    When I give you the full text of an email, do the following:  
    1. Decide if it’s phishing (1) or safe (0).  
    2. Output exactly two lines:  
       • LABEL:<1 or 0>  
       • CONFIDENCE:<a float between 0 and 1>
       • REASON:<comma-separated list of short buzzwords that drove your decision, chosen from:  
         Urgent, Suspicious Link, Generic Greeting, Spoofed Domain, Mismatched URL, Threatening Language,  
         Request for Credentials, Unexpected Attachment, Poor Grammar, Unusual Sender, Hover Discrepancy,  
         Sense of Scarcity, Emotional Manipulation, Link Shortener, Domain Inconsistency, Too Good to Be True, 
         Suspicious Reply-To, Incorrect Branding, Excessive Punctuation, Invoice-Style, Calendar Invite, IP Mismatch  
       >  
    3. Do not add any other text.
        """)

PROMPT_SYSTEM_MACHINE = ("""
    You are a content classification assistant specialized in distinguishing machine-generated text from human-written text.  
    When I give you a piece of text, do the following:  
    1. Decide if it’s machine-generated (1) or human-written (0).  
    2. Output exactly two lines:  
       • LABEL:<1 or 0>  
       • CONFIDENCE:<a float between 0 and 1>
       • REASON:<comma-separated list of short buzzwords that drove your decision, chosen from:  
         Repetitive Phrases, Uniform Punctuation, Overly Formal Tone, Predictable Structure, Lack of Typos,  
         Unnatural Phrasing, Inconsistent Context, Technical Jargon, Generic Content, Personal Anecdotes,  
         Colloquialisms, Slang Usage, Emotional Language, Specific Detail, Unsupported Claims,  
         Domain-Specific Knowledge, Formatting Errors, Abrupt Topic Shifts, Over-Explanation, Under-Explanation  
       >  
    3. Do not output anything else.
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
    conf = None
    text = response.choices[0].message.content.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    m1 = re.search(r"LABEL:\s*<?\s*(\d)\s*>?", lines[0])
    m2 = re.search(r"CONFIDENCE:\s*<?\s*([0-9]*\.?[0-9]+)\s*>?", lines[1])
    m3 = re.search(r"REASON:<?(.+)>?", lines[2])
    try:
        p_label = int(m1.group(1))
    except Exception:
        p_label = None
    try:
        conf = float(m2.group(1)) if m2 else None
        if conf and conf < 0.5: conf = 1-conf
    except Exception:
        p_label = None
    try:
        reasons = [r.strip() for r in m3.group(1).split(",")]
    except Exception:
        reasons = None
    return p_label, conf, reasons



def call_api_and_process_output(mail_text: str, task: str):
    """
    This function calls api and then processes the response
    """
    if task=="Phishing": PROMPT_SYSTEM = PROMPT_SYSTEM_PHISHING
    else: PROMPT_SYSTEM = PROMPT_SYSTEM_MACHINE
    response = api_call(PROMPT_SYSTEM, mail_text)
    label, conf, reasons = split_response(response)
    return label, conf, reasons



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python phishing_api.py '<email_text>'")
        sys.exit(1)
    email = sys.argv[1]
    label, conf, reasons = call_api_and_process_output(email)
    print("P_LABEL=", label)
    print("CONFIDENCE=", conf)
    print("REASONS=", reasons)
