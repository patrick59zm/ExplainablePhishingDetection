import os
import random
import gradio as gr
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv # Import the library
from models.logreg_explainability import get_logreg_mail_specific_explanation, get_logreg_prediction_probabilities
import joblib

from models.xgboost_explainability import get_xgboost_prediction_probabilities, \
    get_xgboost_mail_specific_inherent_explanation

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

from Web_app.get_llm_prediction import call_api_and_process_output
from Web_app.chat_bot import call_chat_bot
from Web_app.get_model_predictions import bert_predict
from Web_app.explanation_processing import explanation_processing


# This function is needed to make xgboost work
def sparse_to_dense_array(sparse_matrix):
    """Converts a sparse matrix to a dense numpy array."""
    if hasattr(sparse_matrix, "toarray"):
        return sparse_matrix.toarray()
    return sparse_matrix


def classify_and_explain_email(raw_email: str, model_name: str, explain_level: str, task: str):
    """Return a random verdict and confidence."""
    # Homogenize the prediction labels
    if task=="Phishing":
        true_label="Phishing"
        false_label="Safe"
    else:
        true_label="Machine Generated"
        false_label="Not Machine Generated"


    if model_name == "Zero-shot SOTA-LLM":
        p_label, confidence, reasons = call_api_and_process_output(raw_email, task)
        verdict = true_label if p_label == 1 else false_label
        explanation = ", ".join(reasons) if reasons else "No explanation"
    elif model_name == "Logistic Regression":
        if task == "Phishing":
            possible_paths = [Path("models/logistic_regression_model_pipeline.joblib")
                                , Path("models\\logistic_regression_model_pipeline.joblib")
                              , Path("../models/logistic_regression_model_pipeline.joblib"),]
        else:
            possible_paths = [Path("models/logistic_regression_model_mgd_pipeline.joblib")
                , Path("models\\logistic_regression_model_mgd_pipeline.joblib")
                , Path("../models/logistic_regression_model_mgd_pipeline.joblib"), ]

        pipeline = None
        for path in possible_paths:
            if path.exists():
                pipeline = joblib.load(path)
                break
        if pipeline is None:
            raise FileNotFoundError("Path not found")

        confidence_info = get_logreg_prediction_probabilities(raw_email, pipeline)

        if confidence_info:
            confidence_details = confidence_info  # Store the full dict
            predicted_class = confidence_info.get("predicted_class")
            if predicted_class == 1:
                verdict = true_label
                confidence =  confidence_details['phishing_probability']
            elif predicted_class == 0:
                verdict = false_label
                confidence = confidence_details['safe_probability']

        explanation_tuples = get_logreg_mail_specific_explanation(raw_email, pipeline)
        formatted_reasons_list = []

        for feature, coeff, tfidf in explanation_tuples:
            formatted_reasons_list.append(f"'{feature}' (Coeff: {coeff:.3f}, TF-IDF: {tfidf:.2f})")
        explanation = ", ".join(formatted_reasons_list)
    elif model_name == "XGBoost":
        if task == "Phishing":
            possible_paths = [Path("models/xgboost_model_pipeline.joblib")
                                , Path("models\\xgboost_model_pipeline.joblib")
                              , Path("../models/xgboost_model_pipeline.joblib"),]
        else:
            possible_paths = [Path("models/xgboost_model_msg_pipeline.joblib")
                , Path("models\\xgboost_model_msg_pipeline.joblib")
                , Path("../models/xgboost_model_msg_pipeline.joblib"), ]

        pipeline = None
        for path in possible_paths:
            if path.exists():
                pipeline = joblib.load(path)
                break
        if pipeline is None:
            raise FileNotFoundError("Path not found")

        confidence_info = get_xgboost_prediction_probabilities(raw_email, pipeline)

        if confidence_info:
            confidence_details = confidence_info  # Store the full dict
            predicted_class = confidence_info.get("predicted_class")
            if predicted_class == 1:
                verdict = true_label
                confidence =  confidence_details['phishing_probability']
            elif predicted_class == 0:
                verdict = false_label
                confidence = confidence_details['safe_probability']

        explanation_tuples = get_xgboost_mail_specific_inherent_explanation(raw_email, pipeline)
        formatted_reasons_list = []

        for feature, coeff, tfidf in explanation_tuples:
            formatted_reasons_list.append(f"'{feature}' (Coeff: {coeff:.3f}, TF-IDF: {tfidf:.2f})")
        explanation = ", ".join(formatted_reasons_list)
    elif model_name[:4] == "BERT":
        
        # Call your bert_predict function directly
        (label_str, conf), expl = bert_predict(raw_email, model_name[5:].lower(),task)
        verdict = true_label if label_str == "phishing" else false_label
        confidence = conf
        explanation = expl
    else:
        verdict = random.choice([false_label, true_label])
        confidence = 0.5
        base = "This is a test explanation."
        details = [f"Detail {explain_level}"]
        explanation = base + " " + "; ".join(details)
    return verdict, confidence, explanation

def detect_and_explain(raw_email: str, model_name: str, explain_level: str, task: str):
    """Classify the email and return stubbed verdict + explanation."""
    verdict, confidence, explanation = classify_and_explain_email(raw_email, model_name, explain_level, task)
    label = f"{verdict.capitalize()} ({confidence * 100:.0f}% confidence)"
    explanation = explanation_processing(explain_level, model_name, explanation, verdict, raw_email)
    return label, explanation

def handle_chat(question, history, raw_email, model_name, explain_level, verdict, explanation):
    """Respond to chat queries, echoing context."""
    context = {"email": raw_email, "model": model_name, "explanation_level": explain_level, "verdict": verdict, "explanation": explanation}
    reply = call_chat_bot(question, history, context)
    history = history or []
    history.append((question, reply))
    return history


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
                    task_selector = gr.Dropdown(
                        label="Choose Task",
                        choices=["Phishing", "Machine Generated"],
                        value="Phishing"
                    )
                    
                    choices = {"Phishing": ["Zero-shot SOTA-LLM", "Logistic Regression", "XGBoost", "BERT-LIME", "BERT-SHAP"],
                              "Machine Generated": ["Logistic Regression", "XGBoost", "BERT-LIME", "BERT-SHAP", "Zero-shot SOTA-LLM"]}
                    def update_second(first_val):
                        d2 = gr.Dropdown(choices[first_val], value=choices[first_val][0], label="Choose Model")
                        return d2 

                    model_selector= update_second("Phishing")

                    task_selector.input(update_second, task_selector,model_selector)
                    
                    # model_selector = gr.Dropdown(
                    #     label="Choose Model",
                    #     choices=choices,
                    #     value="Zero-shot SOTA-LLM"
                    # )
                    
                    explanation_radio = gr.Radio(
                        label="Explanation Type",
                        choices=["Raw XAI output", "Slightly enhanced XAI output", "Greatly simplified explanation"],
                        value="Raw XAI output"
                    )

            analyze_btn.click(
                fn=detect_and_explain,
                inputs=[email_input, model_selector, explanation_radio, task_selector],
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
                inputs=[user_msg, chatbot, email_input, model_selector, explanation_radio, verdict_output, explanation_output],
                outputs=chatbot
            )

demo.launch(server_name="0.0.0.0", server_port=7860)
