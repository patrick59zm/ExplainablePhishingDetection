from openai import OpenAI
import os


api_key = os.getenv("DEEPSEEK_API_KEY")
client  = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def explanation_processing(explanation_type: str, model_type: str, explanation: str, verdict: str, mail: str):
    if explanation_type == "Raw XAI output":
        return explanation
    elif explanation_type == "Slightly enhanced XAI output":
        user_prompt = (
            f"Here is the raw model explanation:\n\n{explanation}\n\n"
            f"Email:\n{mail}\n"
            f"Verdict: {verdict}\n\n"
            "Please rephrase this explanation to be clearer and more concise, "
            "while preserving the original meaning and buzzwords."
        )
    elif explanation_type == "Greatly simplified explanation":
        user_prompt = (
            f"The model use for the verdict and explanation is:\n\n{model_type}\n\n"
            f"Here is the raw model explanation:\n\n{explanation}\n\n"
            f"Email:\n{mail}\n"
            f"Verdict: {verdict}\n\n"
            "Please simplify this explanation so that anyone can understand it, "
            "using plain language and sticking to the core insights."
        )
    else:
        return explanation

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an assistant that reformulates model explanations."},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        stream=False
    )
    return response.choices[0].message.content.strip()
