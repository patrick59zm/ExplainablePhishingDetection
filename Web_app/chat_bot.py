from openai import OpenAI
import os
import re


api_key = os.getenv("DEEPSEEK_API_KEY")
client  = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


PROMPT_SYSTEM = ("""
    You are a cybersecurity assistant. You receive, as context, the full email text plus these classification results:
      • Verdict: “Safe” or “Phishing”
      • Confidence: a number between 0.0 and 1.0 indicating model certainty
      • Explanation: the explanation as the model provides it
      • Model: the name of the model used
      • Explanation Type: the style of explanation requested
    
    When the user asks a question, you should:
      1. Ground your answer strictly in the provided context—do not hallucinate facts.
      2. Refer back to the verdict, confidence, and reasons as needed.
      3. Explain technical terms simply and concisely.
      4. Keep each response brief and to the point—no long essays.
      5. If the user asks for detail beyond what’s in the context, say you don’t have that information.
      6. Be polite, professional, and helpful.
    
    When being asked anything stick to the official verdict, confidence, and reasons. Do not under no circumstances invent anything else. 
    You task is to make sense of it tell why these things led to these results. Be concise and short!
            """)



def call_chat_bot(question: str, history: list, context: dict) -> list:
    messages = [{"role": "system", "content": PROMPT_SYSTEM}]

    # Classification context as its own system-level message
    context_content = (f"Email:\n{context['email']}\n" f"Model: {context['model']}\n" 
                       f"Official Verdict: {context.get('verdict')}\n" f"Official Explanation: {context.get('explanation')}\n")

    messages.append({"role": "system", "content": context_content})

    # Combine prior turns into one contextual block
    for user_turn, bot_turn in history or []:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": bot_turn})

    # 4) Append the new question as a fresh user message
    messages.append({"role": "user", "content": question})
         # Call the LLM
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=300,
        stream=False
    )
    return response.choices[0].message.content.strip()





