import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from joblib import dump

def sparse_to_dense_array(sparse_matrix):
    """Converts a sparse matrix to a dense numpy array."""
    return sparse_matrix.toarray()


def train_xgboost_model(train_data_path: str, test_data_path: str,
                       max_features: int = 20000,
                       max_depth: int = 6, learning_rate: float = 0.1,
                       n_estimators: int = 1300,
                       plot_importance: bool = True,
                       frac: float = 1.00) -> tuple[XGBClassifier, TfidfVectorizer]:
    """
    Trains an XGBoost model to classify emails as phishing or safe, using unigrams and bigrams.

    Args:
        train_data_path (str): Path to the training data CSV file.
        test_data_path (str): Path to the test data CSV file.
        max_features (int, optional): Maximum number of features to use from the TF-IDF vectorization. Defaults to 20000.
        max_depth (int, optional): Maximum depth of the XGBoost trees. Defaults to 6.
        learning_rate (float, optional): Learning rate for XGBoost. Defaults to 0.1.
        n_estimators (int, optional): Number of trees in the XGBoost ensemble. Defaults to 800.
        plot_importance (bool, optional): Whether to plot the feature importance. Defaults to True.
        frac (float, optional): Fraction of dataset to use. Defaults to 1.00.

    Returns:
        xgb.XGBClassifier: The trained XGBoost model.
    """
    # 1. Data Loading
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    df_train = df_train[["p_label", "sterilized_text"]]
    df_test = df_test[["p_label", "sterilized_text"]]

    df_train.rename(columns={'p_label': 'Email Type', 'sterilized_text': 'Email Text'}, inplace=True)
    df_test.rename(columns={'p_label': 'Email Type', 'sterilized_text': 'Email Text'}, inplace=True)

    df_train['Email Text'] = df_train['Email Text'].fillna('')
    df_test['Email Text'] = df_test['Email Text'].fillna('')

    df_train = df_train.sample(frac=frac, random_state=42)
    df_test = df_test.sample(frac=frac, random_state=42)

    # Combine training and testing data for consistent preprocessing and vectorization
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Ensure the input DataFrame has the necessary columns
    if "Email Text" not in df.columns or "Email Type" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain 'Email Text' and 'Email Type' columns."
        )

    # 2. Feature Extraction
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["Email Text"])  # Use the preprocessed column
    y = df["Email Type"]

    #3. Split data
    X_train = vectorizer.transform(df_train["Email Text"])
    X_test = vectorizer.transform(df_test["Email Text"])
    y_train = df_train["Email Type"]
    y_test = df_test["Email Type"]

    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    # 4. Model Training
    # Model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
    )

    model.fit(X_train_dense, y_train)

    # 5. Model Evaluation
    y_pred_bigrams = model.predict(X_test_dense)
    accuracy_bigrams = accuracy_score(y_test, y_pred_bigrams)
    print("XGBoost Results:")
    print(f"Model Accuracy: {accuracy_bigrams:.4f}")
    print(classification_report(y_test, y_pred_bigrams))

    # 6. Feature Importance
    if plot_importance:
        # Extract feature importance as a dictionary
        importance_dict = model.get_booster().get_score(importance_type="gain")

        # Convert to DataFrame
        importance_df = pd.DataFrame(
            list(importance_dict.items()), columns=["Feature", "Importance"]
        )

        importance_df["Feature"] = importance_df["Feature"].apply(
            lambda x: int(x[1:])
        )  # Remove 'f' prefix and convert to int

        # Sort by importance
        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(
            30
        )  # Top 30

        # Map feature indices back to words
        feature_names = vectorizer.get_feature_names_out()  # Get actual words
        importance_df["Feature"] = importance_df["Feature"].map(
            lambda i: feature_names[i] if i < len(feature_names) else f"Unknown_{i}"
        )

        plt.figure(figsize=(12, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="royalblue")
        plt.xlabel("Gain")
        plt.ylabel("Feature (Word)")
        plt.title("Top 30 Most Important Words for Phishing Detection")
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.show()

    return model, vectorizer  # Return both the trained model and the fitted vectorizer



def retrain_xgboost_model(df: pd.DataFrame, vectorizer: TfidfVectorizer,
                         max_depth: int = 6, learning_rate: float = 0.1,
                         n_estimators: int = 800) -> Pipeline:
    """
    Retrains an XGBoost model on the entire dataset.

    Args:
        df (pd.DataFrame): The input DataFrame containing email text and labels.
        vectorizer (TfidfVectorizer): The fitted TfidfVectorizer from the training phase.
        max_depth (int, optional): Maximum depth of the XGBoost trees. Defaults to 6.
        learning_rate (float, optional): Learning rate for XGBoost. Defaults to 0.1.
        n_estimators (int, optional): Number of trees in the XGBoost ensemble. Defaults to 800.

    Returns:
        xgb.XGBClassifier: The retrained XGBoost model.
    """
    df = df[["p_label", "sterilized_text"]]
    df.rename(columns={'p_label': 'Email Type', 'sterilized_text': 'Email Text'}, inplace=True)
    df['Email Text'] = df['Email Text'].fillna('')


    X = df["Email Text"]  # Use the fitted vectorizer
    y = df["Email Type"]

    # X_dense = X.toarray()

    # Model
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
    )

    dense_transformer = FunctionTransformer(sparse_to_dense_array, accept_sparse=True,validate=False,  check_inverse=False)

    pipeline_xgboost = Pipeline([
        ("tfidf", vectorizer),
        ('dense', dense_transformer),
        ("logreg", model)
    ])

    pipeline_xgboost.fit(X, y)  # Train on the full dataset

    return pipeline_xgboost


if __name__ == "__main__":
    train_data_path = "../data/train/train_dataset.csv"
    test_data_path = "../data/test/test_dataset.csv"

    # Train the model
    trained_model, fitted_vectorizer = train_xgboost_model(train_data_path, test_data_path, frac=0.05)

    # Load the entire dataset for retraining
    df_full = pd.concat([pd.read_csv(train_data_path), pd.read_csv(test_data_path)], ignore_index=True)
    df_full = df_full.sample(frac=0.05, random_state=42)

    # Retrain the model on the full dataset
    final_model = retrain_xgboost_model(df_full, fitted_vectorizer)

    dump(final_model, "xgboost_model_pipeline.joblib")
    print("\nFinal model retrained and saved.")
