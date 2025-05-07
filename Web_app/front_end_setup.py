import random
import gradio as gr
from Web_app.get_llm_prediction import call_api_and_process_output
from models.logreg_explainability import get_logreg_mail_specific_explanation
import joblib


def classify_and_explain_email(raw_email: str, model_name: str, explain_level: str):
    """Return a random verdict and confidence."""
    if model_name == "Zero-shot SOTA-LLM":
        p_label, confidence, reasons = call_api_and_process_output(raw_email)
        verdict = "Phishing" if p_label == 1 else "Safe"
        explanation = ", ".join(reasons) if reasons else "No explanation"
    elif model_name == "logreg":
        verdict, confidence, reasons = get_logreg_mail_specific_explanation(raw_email, pipeline=joblib.load("models/logistic_regression_model_pipeline.joblib"))
        explanation = ", ".join(reasons) if reasons else "No explanation"
    else:
        verdict = random.choice(["safe", "phishing"])
        confidence = random.random()
        base = "This is a test explanation."
        details = [f"Detail {explain_level}"]
        explanation = base + " " + "; ".join(details)
    return verdict, confidence, explanation

def chat_response(question, history, context):
    """Return a canned chat response incorporating context."""
    snippet = context.get("email", "<no email>")[:30].replace("\n", " ") + "..."
    question = f'Your question: "{question}"'
    reply = f"Reply: '{question}'"
    history = history or []
    history.append((question, reply))
    return history

def detect_and_explain(raw_email: str, model_name: str, explain_level: str):
    """Classify the email and return stubbed verdict + explanation."""
    verdict, confidence, explanation = classify_and_explain_email(raw_email, model_name, explain_level)
    label = f"{verdict.capitalize()} ({confidence * 100:.0f}% confidence)"
    return label, explanation

def handle_chat(question, history, raw_email, model_name, explain_level):
    """Respond to chat queries, echoing context."""
    context = {
        "email": raw_email,
        "model": model_name,
        "explanation_level": explain_level
    }
    return chat_response(question, history, context)




with gr.Blocks(theme="default") as demo:
    gr.Markdown("# Phishing Detector")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("Email Analysis"):
                    email_input = gr.Textbox(lines=8, label="Paste the email you want to evaluate here…",
                                                placeholder="Headers, body, links, etc.")
                    analyze_btn = gr.Button("Analyze")
                    verdict_output = gr.Label(label="Verdict")
                    explanation_output = gr.Textbox(label="Reasoning:",
                                                    placeholder="The reasoning will show up here", lines=6)

                with gr.TabItem("Settings"):
                    model_selector = gr.Dropdown(
                        label="Choose Model",
                        choices=["Zero-shot SOTA-LLM", "logreg", "modelC"],
                        value="Zero-shot SOTA-LLM"
                    )
                    explanation_radio = gr.Radio(
                        label="Explanation Type",
                        choices=["Raw XAI output", "Enhance Explanation", "Simplified Explanation"],
                        value="Simplified Explanation"
                    )

            analyze_btn.click(
                fn=detect_and_explain,
                inputs=[email_input, model_selector, explanation_radio],
                outputs=[verdict_output, explanation_output]
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Assistant")
            user_msg = gr.Textbox(
                label="Ask about the last scan…",
                placeholder="Why is this suspicious?"
            )
            user_msg.submit(
                fn=handle_chat,
                inputs=[user_msg, chatbot, email_input, model_selector, explanation_radio],
                outputs=chatbot
            )

demo.launch(server_name="0.0.0.0", server_port=7860)
