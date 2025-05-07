# app.py
# Simple Gradio app for phishing detection with chat
# Install dependencies:
#   pip install gradio
# Run:
#   python app.py

import gradio as gr
from your_model import classify_email, explain_email
from your_chatbot import chat_response


def detect_and_explain(raw_email: str):
    """
    Classify the email and return a verdict label plus explanation.
    Replace classify_email and explain_email with your actual model calls.
    """
    verdict, confidence = classify_email(raw_email)
    explanation = explain_email(raw_email)
    label = f"{verdict.capitalize()} ({confidence * 100:.0f}% confidence)"
    return label, explanation


def handle_chat(question, history):
    """
    Respond to chat queries using your LLM-powered chat_response.
    Preloads context from the last classification.
    """
    return chat_response(question, history)


with gr.Blocks() as demo:
    gr.Markdown("# Phishing Detector")

    with gr.Tab("Email Scan"):
        email_input = gr.Textbox(
            lines=8,
            label="Paste full email here…",
            placeholder="Headers, body, links, etc."
        )
        analyze_btn = gr.Button("Analyze")
        verdict_output = gr.Label(label="Verdict")
        explanation_output = gr.Textbox(label="Why?", lines=6)
        analyze_btn.click(
            fn=detect_and_explain,
            inputs=email_input,
            outputs=[verdict_output, explanation_output]
        )

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        user_msg = gr.Textbox(
            label="Ask about the last scan…",
            placeholder="Why is this suspicious?"
        )
        user_msg.submit(
            fn=handle_chat,
            inputs=[user_msg, chatbot],
            outputs=chatbot
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
