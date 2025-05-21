# Phishing Detection Web App

This directory contains a self‑contained web application that lets you interactively classify emails as phishing or safe, and explore human‑readable explanations for the model’s verdicts.

## Folder Content

* **`__init__.py`** – Marks this folder as a Python package.
* **`chat_bot.py`** – Entrypoint for the web server; defines routes and launches the app.
* **`front_end_setup.py`** – Bundles and builds front‑end assets (JS/CSS) before you start the server.
* **`get_model_predictions.py`** – Wraps your locally trained models (logistic regression, XGBoost) to produce verdicts and raw XAI outputs.
* **`get_llm_prediction.py`** – Queries the DeepSeek LLM API to classify and explain emails using an LLM.
* **`explanation_processing.py`** – Post‑processes raw explanations into one of three flavors
* **`README.md`** – (This file.)

---

To use the web_app you have to do several things:

1. **Set your DeepSeek API key**

   The LLM‑based explanation and classification rely on DeepSeek’s API. Before building or running the app, export your key:

   ```bash
   export DEEPSEEK_API_KEY="sk-<your-key>"
   ```

2. **Download and put other models into the right place**

...


3. **Build the front end**

   This compiles any JavaScript/CSS needed by the UI:

   ```bash
   python3 -m Web_app.front_end_setup
   ```

5. **Open in your browser**

   Visit `http://localhost:7860` (or the port printed in your console) to start classifying emails.

---

## How It Works

1. **User Settings**
   * Chose what you want to detect:
     * Phishing
     * Machine generated mails
   * Choose one of the available models:

     * **Local models**: logistic regression, XGBoost, Bert-Lime, Bert-Shap.
     * **LLM model**: DeepSeek’s `deepseek-chat` endpoint.
   * Pick an explanation style:

     * *Raw XAI output* (exact model explanation)
     * *Slightly enhanced* (concise rephrase)
     * *Greatly simplified* (plain‑language summary)

2. **User interaction**
    1. Paste a mail into the box on the top left
    2. Click analyze
   3. A verdict and confidence of the verdict are displayed and the explanation of it
   4. If interested ask the chat-bot for more details about the classification
---

## Tips & Troubleshooting

* **Missing API key?** You’ll see authentication errors when calling DeepSeek. Double‑check `DEEPSEEK_API_KEY` is set.
* **Rebuild front end** after changing any UI files: `python3 -m Web_app.front_end_setup`.
* **Logs** for classification and explanation steps are printed to the console if you run in debug mode.

---

## 📜 License

This code is released under the same license as the top‑level project. See the root `LICENSE` file for details.
