from pathlib import Path

import joblib

DEFAULT_LOGREG_PIPELINE_PATH = "logistic_regression_model_pipeline.joblib"


def load_logistic_regression_pipeline(pipeline_path=DEFAULT_LOGREG_PIPELINE_PATH):
    """
    Loads the trained scikit-learn pipeline for Logistic Regression.
    This pipeline is expected to contain TF-IDF, scaler, and the model.

    Args:
        pipeline_path (str): Path to the saved pipeline model (.joblib).

    Returns:
        sklearn.pipeline.Pipeline: The loaded pipeline, or None if loading fails.
    """
    try:
        loaded_pipeline = joblib.load(pipeline_path)
        print(f"Logistic Regression pipeline loaded from: {pipeline_path}")
        if not all(step in loaded_pipeline.named_steps for step in ['tfidf', 'scaler', 'logreg']):
            print("Warning: Loaded pipeline does not seem to have the expected steps: 'tfidf', 'scaler', 'logreg'.")
        return loaded_pipeline
    except FileNotFoundError:
        print(f"Error: Logistic Regression pipeline file not found. Searched for:")
        print(f"  Pipeline: {Path(pipeline_path).resolve()}")
        print("Please ensure the path is correct and the file exists.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the Logistic Regression pipeline: {e}")
        return None


# --- 2. Get Prediction Probabilities ---
def get_logreg_prediction_probabilities(email_text, pipeline):
    """
    Gets the phishing and non-phishing probabilities for a given email text
    using the Logistic Regression pipeline.

    Args:
        email_text (str): The raw text of the email.
        pipeline (sklearn.pipeline.Pipeline): The loaded scikit-learn pipeline.

    Returns:
        dict: A dictionary with 'safe_probability' and 'phishing_probability',
              or None if prediction fails.
              Assumes class 0 is "safe" and class 1 is "phishing".
    """
    if not pipeline:
        print("Error: Logistic Regression pipeline not loaded.")
        return None
    try:
        probabilities = pipeline.predict_proba([email_text])

        safe_prob = probabilities[0, 0]
        phishing_prob = probabilities[0, 1]

        return {
            "safe_probability": safe_prob,
            "phishing_probability": phishing_prob,
            "predicted_class": 1 if phishing_prob > safe_prob else 0
        }
    except Exception as e:
        print(f"Error during Logistic Regression probability prediction: {e}")
        return None


def get_logreg_general_explainability(pipeline, top_n=20):
    """
    Provides global feature importances (coefficients) from the Logistic Regression model
    in the pipeline.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The loaded scikit-learn pipeline.
        top_n (int): The number of top features (by absolute coefficient value) to display.

    Returns:
        list: A list of (feature_name, coefficient_value) tuples for the top_n features,
              sorted by absolute coefficient value, or None if an error occurs.
    """
    if not pipeline:
        print("Error: Logistic Regression pipeline not loaded.")
        return None
    try:
        logreg_model = pipeline.named_steps['logreg']
        vectorizer = pipeline.named_steps['tfidf']

        if not hasattr(logreg_model, 'coef_'):
            print("Error: The 'logreg' step in the pipeline does not have 'coef_'.")
            return None
        if not hasattr(vectorizer, 'get_feature_names_out'):
            print("Error: The 'tfidf' step in the pipeline cannot provide feature names.")
            return None

        coefficients = logreg_model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()

        feature_importance_map = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)

        return feature_importance_map[:top_n]

    except KeyError as e:
        print(f"Error: Step '{e}' not found in the pipeline. Check pipeline step names (expected 'tfidf', 'logreg').")
        return None
    except Exception as e:
        print(f"Error getting general Logistic Regression explainability: {e}")
        return None


def get_logreg_mail_specific_explanation(email_text, pipeline=joblib.load("logistic_regression_model_pipeline.joblib"), top_n_instance=15):
    """
    Provides an inherent explanation for a specific email by showing the
    coefficients of the features (words/n-grams) present in that email.

    Args:
        email_text (str): The raw text of the email.
        pipeline (sklearn.pipeline.Pipeline): The loaded scikit-learn pipeline.
        top_n_instance (int): Max number of features from the email to show, sorted by abs coefficient.


    Returns:
        list: A list of (feature_name, coefficient_value, tfidf_score) tuples for features
              present in the email, sorted by absolute coefficient value. Returns None on error.
    """
    if not pipeline:
        print("Error: Logistic Regression pipeline not loaded.")
        return None
    try:
        vectorizer = pipeline.named_steps['tfidf']
        logreg_model = pipeline.named_steps['logreg']

        if not hasattr(logreg_model, 'coef_'):
            print("Error: The 'logreg' step in the pipeline does not have 'coef_'.")
            return None
        if not hasattr(vectorizer, 'get_feature_names_out'):
            print("Error: The 'tfidf' step in the pipeline cannot provide feature names.")
            return None

        all_feature_names = vectorizer.get_feature_names_out()
        all_coefficients = logreg_model.coef_[0]
        feature_to_coeff_map = dict(zip(all_feature_names, all_coefficients))

        tfidf_vector = vectorizer.transform([email_text])

        present_feature_indices = tfidf_vector.indices
        present_feature_tfidf_scores = tfidf_vector.data

        instance_explanation = []
        for i, feature_idx in enumerate(present_feature_indices):
            feature_name = all_feature_names[feature_idx]
            coefficient = feature_to_coeff_map.get(feature_name, 0.0)  # Should always be found
            tfidf_score = present_feature_tfidf_scores[i]
            instance_explanation.append((feature_name, coefficient, tfidf_score))

        instance_explanation.sort(key=lambda x: abs(x[1]), reverse=True)

        return instance_explanation[:top_n_instance]

    except KeyError as e:
        print(f"Error: Step '{e}' not found in the pipeline. Check pipeline step names (expected 'tfidf', 'logreg').")
        return None
    except Exception as e:
        print(f"Error getting mail-specific Logistic Regression explanation: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Explain Logistic Regression Model Script ---")

    loaded_logreg_pipeline = load_logistic_regression_pipeline()

    if loaded_logreg_pipeline:
        print("\n--- 1. General Model Explainability (Logistic Regression Coefficients) ---")
        general_explanations_logreg = get_logreg_general_explainability(loaded_logreg_pipeline, top_n=10)
        if general_explanations_logreg:
            print("Top 10 global feature coefficients (by absolute value):")
            for feature, coefficient in general_explanations_logreg:
                print(f"- Feature: '{feature}', Coefficient: {coefficient:.4f}")

        print("\n--- 2. Specific Email Analysis (Logistic Regression) ---")
        sample_phishing_email = "URGENT call now to claim your free iPhone prize! Limited offer! http://phish.com/prize click here"
        sample_safe_email = "Hi John, Here are the documents you requested for our meeting next week. Regards, Sarah"

        for i, email_text_to_analyze in enumerate([sample_phishing_email, sample_safe_email]):
            print(f"\n--- Analyzing Email {i + 1} (Logistic Regression) ---")
            print(f"Email Text: \"{email_text_to_analyze[:100]}...\"")

            probabilities = get_logreg_prediction_probabilities(email_text_to_analyze, loaded_logreg_pipeline)
            if probabilities:
                print(f"  Predicted Class: {'Phishing' if probabilities['predicted_class'] == 1 else 'Safe'}")
                print(f"  Safe Probability: {probabilities['safe_probability']:.4f}")
                print(f"  Phishing Probability: {probabilities['phishing_probability']:.4f}")

            print("  Getting inherent explanation for this email (features present and their coefficients)...")
            specific_explanation = get_logreg_mail_specific_explanation(email_text_to_analyze, loaded_logreg_pipeline,
                                                                        top_n_instance=10)
            if specific_explanation:
                print(
                    f"  Top {len(specific_explanation)} features in this email contributing to prediction (sorted by abs. coefficient):")
                for feature, coeff, tfidf in specific_explanation:
                    print(f"  - Feature: '{feature}', Model Coefficient: {coeff:.4f}, TF-IDF in email: {tfidf:.4f}")
            else:
                print("  Could not generate specific explanation for this email.")
    else:
        print("\nCould not run examples because Logistic Regression pipeline failed to load.")

    print("\n--- Script Finished ---")