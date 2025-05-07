from pathlib import Path  # For easier path management in example

import joblib
import xgboost as xgb  # Required for loading pipeline with XGBoost model


def sparse_to_dense_array(sparse_matrix):
    """Converts a sparse matrix to a dense numpy array."""
    if hasattr(sparse_matrix, "toarray"):
        return sparse_matrix.toarray()
    return sparse_matrix


DEFAULT_XGBOOST_PIPELINE_PATH = "xgboost_model_pipeline.joblib"


def load_xgboost_pipeline(pipeline_path=DEFAULT_XGBOOST_PIPELINE_PATH):
    """
    Loads the trained scikit-learn pipeline for XGBoost.
    This pipeline is expected to contain TF-IDF, a dense transformer, and the XGBoost model.

    Args:
        pipeline_path (str): Path to the saved pipeline model (.joblib).

    Returns:
        sklearn.pipeline.Pipeline: The loaded pipeline, or None if loading fails.
    """
    try:
        loaded_pipeline = joblib.load(pipeline_path)
        print(f"XGBoost pipeline loaded from: {pipeline_path}")
        expected_steps = ['tfidf', 'dense', 'logreg']
        if not all(step in loaded_pipeline.named_steps for step in expected_steps):
            print(f"Warning: Loaded pipeline does not seem to have all expected steps: {expected_steps}.")
            print(f"Found steps: {list(loaded_pipeline.named_steps.keys())}")
        return loaded_pipeline
    except FileNotFoundError:
        print(f"Error: XGBoost pipeline file not found. Searched for:")
        print(f"  Pipeline: {Path(pipeline_path).resolve()}")
        print("Please ensure the path is correct and the file exists.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the XGBoost pipeline: {e}")
        return None


def get_xgboost_prediction_probabilities(email_text, pipeline):
    """
    Gets the phishing and non-phishing probabilities for a given email text
    using the XGBoost pipeline.

    Args:
        email_text (str): The raw text of the email.
        pipeline (sklearn.pipeline.Pipeline): The loaded scikit-learn pipeline.

    Returns:
        dict: A dictionary with 'safe_probability' and 'phishing_probability',
              or None if prediction fails.
              Assumes class 0 is "safe" and class 1 is "phishing".
    """
    if not pipeline:
        print("Error: XGBoost pipeline not loaded.")
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
        print(f"Error during XGBoost probability prediction: {e}")
        return None


def get_xgboost_general_explainability(pipeline, top_n=20, importance_type='gain'):
    """
    Provides global feature importances from the XGBoost model in the pipeline.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The loaded scikit-learn pipeline.
        top_n (int): The number of top features to display.
        importance_type (str): Type of importance to get ('weight', 'gain', 'cover', 'total_gain', 'total_cover').
                               'gain' is a common default.

    Returns:
        list: A list of (feature_name, importance_score) tuples for the top_n features,
              sorted by importance, or None if an error occurs.
    """
    if not pipeline:
        print("Error: XGBoost pipeline not loaded.")
        return None
    try:
        xgb_model_step_name = 'logreg'
        if xgb_model_step_name not in pipeline.named_steps:
            # Fallback if the user named it differently, e.g. 'xgb_classifier'
            potential_names = [name for name in pipeline.named_steps if
                               isinstance(pipeline.named_steps[name], xgb.XGBClassifier)]
            if not potential_names:
                print(
                    f"Error: XGBoost classifier step not found in pipeline. Looked for '{xgb_model_step_name}' and other XGBClassifier instances.")
                return None
            xgb_model_step_name = potential_names[0]
            print(f"Info: Using XGBoost model step named '{xgb_model_step_name}'.")

        xgb_model = pipeline.named_steps[xgb_model_step_name]
        vectorizer = pipeline.named_steps['tfidf']

        if not hasattr(xgb_model, 'get_booster'):
            print(
                f"Error: The '{xgb_model_step_name}' step in the pipeline is not a standard XGBoost model with get_booster().")
            return None
        if not hasattr(vectorizer, 'get_feature_names_out'):
            print("Error: The 'tfidf' step in the pipeline cannot provide feature names.")
            return None

        booster = xgb_model.get_booster()
        importance_scores = booster.get_score(importance_type=importance_type)  # dict: {'f0': score, 'f1': score, ...}

        all_feature_names = vectorizer.get_feature_names_out()

        feature_importance_map = []
        for f_idx_str, score in importance_scores.items():
            try:
                # XGBoost features are named 'f0', 'f1', ...
                feature_index = int(f_idx_str[1:])
                if feature_index < len(all_feature_names):
                    feature_name = all_feature_names[feature_index]
                    feature_importance_map.append((feature_name, score))
                else:
                    print(
                        f"Warning: Feature index {feature_index} out of bounds for feature names array (len: {len(all_feature_names)}).")
            except ValueError:
                print(f"Warning: Could not parse feature index from '{f_idx_str}'.")

        feature_importance_map.sort(key=lambda x: x[1], reverse=True)

        return feature_importance_map[:top_n]

    except KeyError as e:
        print(
            f"Error: Step '{e}' not found in the pipeline. Check pipeline step names (expected 'tfidf', 'dense', '{xgb_model_step_name}').")
        return None
    except Exception as e:
        print(f"Error getting general XGBoost explainability: {e}")
        return None


# --- 4. Get Mail-Specific Inherent Explanation (Global Importances of Present Words) ---
def get_xgboost_mail_specific_inherent_explanation(email_text, pipeline, top_n_instance=15, importance_type='gain'):
    """
    Provides an inherent explanation for a specific email by showing the
    global importance scores of the features (words/n-grams) present in that email.
    Note: This is not a true local explanation like SHAP, but shows which globally
    important features are active in this specific email.

    Args:
        email_text (str): The raw text of the email.
        pipeline (sklearn.pipeline.Pipeline): The loaded scikit-learn pipeline.
        top_n_instance (int): Max number of features from the email to show, sorted by global importance.
        importance_type (str): Type of importance to use ('weight', 'gain', 'cover').

    Returns:
        list: A list of (feature_name, global_importance_score, tfidf_score_in_email) tuples for
              features present in the email, sorted by global importance. Returns None on error.
    """
    if not pipeline:
        print("Error: XGBoost pipeline not loaded.")
        return None
    try:
        vectorizer = pipeline.named_steps['tfidf']
        # Based on your training code, the XGBoost model step is named 'logreg'
        xgb_model_step_name = 'logreg'
        if xgb_model_step_name not in pipeline.named_steps:
            potential_names = [name for name in pipeline.named_steps if
                               isinstance(pipeline.named_steps[name], xgb.XGBClassifier)]
            if not potential_names:
                print(
                    f"Error: XGBoost classifier step not found in pipeline. Looked for '{xgb_model_step_name}' and other XGBClassifier instances.")
                return None
            xgb_model_step_name = potential_names[0]

        xgb_model = pipeline.named_steps[xgb_model_step_name]

        if not hasattr(xgb_model, 'get_booster') or not hasattr(vectorizer, 'get_feature_names_out'):
            print("Error: Pipeline components missing required methods (get_booster or get_feature_names_out).")
            return None

        # 1. Get global feature importances from the model
        booster = xgb_model.get_booster()
        global_importance_scores = booster.get_score(importance_type=importance_type)  # dict: {'f0': score, ...}
        all_feature_names = vectorizer.get_feature_names_out()

        # Create a map from actual feature name to global importance score
        feature_to_global_importance_map = {}
        for f_idx_str, score in global_importance_scores.items():
            try:
                feature_index = int(f_idx_str[1:])
                if feature_index < len(all_feature_names):
                    feature_to_global_importance_map[all_feature_names[feature_index]] = score
            except ValueError:
                pass  # Ignore if f_idx_str is not in 'fX' format

        # 2. Transform the specific email text to get its TF-IDF scores
        tfidf_vector = vectorizer.transform([email_text])

        # 3. Identify features present in this specific email
        present_feature_indices = tfidf_vector.indices
        present_feature_tfidf_scores = tfidf_vector.data

        instance_explanation = []
        for i, feature_idx in enumerate(present_feature_indices):
            feature_name = all_feature_names[feature_idx]
            global_importance = feature_to_global_importance_map.get(feature_name, 0.0)  # Get its global importance
            tfidf_score = present_feature_tfidf_scores[i]
            if global_importance > 0:  # Only consider features that had some global importance
                instance_explanation.append((feature_name, global_importance, tfidf_score))

        # Sort by global importance value to see most impactful words (globally) in this email
        instance_explanation.sort(key=lambda x: x[1], reverse=True)

        return instance_explanation[:top_n_instance]

    except KeyError as e:
        print(f"Error: Step '{e}' not found in the pipeline. Check pipeline step names.")
        return None
    except Exception as e:
        print(f"Error getting mail-specific XGBoost inherent explanation: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Explain XGBoost Model Script (Inherent Explainability) ---")

    # Dummy file creation for testing if the actual model isn't available:
    # try:
    #     joblib.load(DEFAULT_XGBOOST_PIPELINE_PATH)
    # except FileNotFoundError:
    #     print(f"Creating dummy {DEFAULT_XGBOOST_PIPELINE_PATH} for testing...")
    #     dummy_tfidf = TfidfVectorizer()
    #     dummy_dense_transformer = FunctionTransformer(sparse_to_dense_array, accept_sparse=True, validate=False, check_inverse=False)
    #     dummy_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    #     sample_texts = ["example text with important keyword", "another example", "safe email with offer"]
    #     sample_labels = [1, 1, 0]

    #     # Fit the dummy pipeline (this is a bit contrived for a dummy but necessary for pipeline structure)
    #     dummy_pipeline_for_saving = Pipeline([
    #         ('tfidf', dummy_tfidf),
    #         ('dense', dummy_dense_transformer),
    #         ('logreg', dummy_xgb) # Using 'logreg' as per user's training script
    #     ])
    #     dummy_pipeline_for_saving.fit(sample_texts, sample_labels)
    #     joblib.dump(dummy_pipeline_for_saving, DEFAULT_XGBOOST_PIPELINE_PATH)
    # print("-" * 30)

    loaded_xgb_pipeline = load_xgboost_pipeline()

    if loaded_xgb_pipeline:
        print("\n--- 1. General Model Explainability (XGBoost Feature Importances) ---")
        general_explanations_xgb = get_xgboost_general_explainability(loaded_xgb_pipeline, top_n=10,
                                                                      importance_type='gain')
        if general_explanations_xgb:
            print("Top 10 global feature importances (by 'gain'):")
            for feature, importance in general_explanations_xgb:
                print(f"- Feature: '{feature}', Importance (Gain): {importance:.4f}")

        print("\n--- 2. Specific Email Analysis (XGBoost) ---")
        sample_phishing_email = "URGENT call now to claim your free iPhone prize! Limited offer! http://phish.com/prize click here"
        sample_safe_email = "Hi John, Here are the documents you requested for our meeting next week. Regards, Sarah"

        for i, email_text_to_analyze in enumerate([sample_phishing_email, sample_safe_email]):
            print(f"\n--- Analyzing Email {i + 1} (XGBoost) ---")
            print(f"Email Text: \"{email_text_to_analyze[:100]}...\"")

            probabilities = get_xgboost_prediction_probabilities(email_text_to_analyze, loaded_xgb_pipeline)
            if probabilities:
                print(f"  Predicted Class: {'Phishing' if probabilities['predicted_class'] == 1 else 'Safe'}")
                print(f"  Safe Probability: {probabilities['safe_probability']:.4f}")
                print(f"  Phishing Probability: {probabilities['phishing_probability']:.4f}")

            print("  Getting inherent explanation for this email (globally important features present)...")
            specific_explanation = get_xgboost_mail_specific_inherent_explanation(email_text_to_analyze,
                                                                                  loaded_xgb_pipeline,
                                                                                  top_n_instance=10)
            if specific_explanation:
                print(
                    f"  Top {len(specific_explanation)} globally important features present in this email (sorted by global importance):")
                for feature, global_importance, tfidf in specific_explanation:
                    print(
                        f"  - Feature: '{feature}', Global Importance (Gain): {global_importance:.4f}, TF-IDF in email: {tfidf:.4f}")
            else:
                print("  Could not generate specific inherent explanation for this email.")
    else:
        print("\nCould not run examples because XGBoost pipeline failed to load.")

    print("\n--- Script Finished ---")
