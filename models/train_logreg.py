from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logistic_regression_model(
    train_data_path: str,
    test_data_path: str,
    max_features: int = 20000,
    ngram_range: tuple = (1, 2),
    solver: str = "liblinear",
    C: float = 1.0,
    plot_coefficients: bool = True,
) -> tuple[Any, TfidfVectorizer]:
    """
    Trains a Logistic Regression model to classify emails as phishing or safe.
    Saves the trained model and fitted vectorizer to files.

    Args:
        train_data_path (str): Path to the training data CSV file.
        test_data_path (str): Path to the test data CSV file.
        max_features (int, optional): Maximum number of features to use from the TF-IDF vectorization. Defaults to 20000.
        ngram_range (tuple, optional):  Defaults to (1, 2).
        solver (str, optional): Solver to use for Logistic Regression.  Defaults to 'liblinear'.
        C (float, optional): Regularization parameter. Defaults to 1.0
        plot_coefficients (bool, optional): Whether to plot the coefficients. Defaults to True.

    Returns:
        LogisticRegression: The trained Logistic Regression model.
    """
    # 1. Data Loading
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    df_train = df_train[["p_label", "sterilized_text"]]
    df_test = df_test[["p_label", "sterilized_text"]]

    df_train.rename(columns={'p_label': 'Email Type', 'sterilized_text': 'Email Text'}, inplace=True)
    df_test.rename(columns={'p_label': 'Email Type', 'sterilized_text': 'Email Text'}, inplace=True)

    # Use the column names 'Email Text' and 'Email Type' as in the original notebook
    df_train = df_train[["Email Type", "Email Text"]]
    df_test = df_test[["Email Type", "Email Text"]]

    df_train['Email Text'] = df_train['Email Text'].fillna('')
    df_test['Email Text'] = df_test['Email Text'].fillna('')

    # Combine training and testing data for consistent preprocessing and vectorization
    df = pd.concat([df_train, df_test], ignore_index=True)

    # 2. Feature Extraction
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(df["Email Text"])  # Use the preprocessed column
    y = df["Email Type"]

    # 3. Data Splitting
    X_train = vectorizer.transform(df_train["Email Text"])
    X_test = vectorizer.transform(df_test["Email Text"])
    y_train = df_train["Email Type"]
    y_test = df_test["Email Type"]

    # 4. Model Training
    # Logistic Regression with scaling
    model = LogisticRegression(solver=solver, random_state=42, C=C)
    pipeline_logistic = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", model)
    ])
    pipeline_logistic.fit(X_train, y_train)
    y_pred_logistic = pipeline_logistic.predict(X_test)
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    print("Logistic Regression Results:")
    print(f"Model Accuracy: {accuracy_logistic:.4f}")
    print(classification_report(y_test, y_pred_logistic))

    # 5. Feature Importance (Coefficients)
    if plot_coefficients:
        logreg_model = pipeline_logistic.named_steps[
            "logreg"
        ]
        coefficients = logreg_model.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Coefficient": coefficients}
        )
        feature_importance_df["Abs_Coefficient"] = np.abs(coefficients)
        feature_importance_df = feature_importance_df.sort_values(
            by="Abs_Coefficient", ascending=False
        )
        print("\nTop 40 Most Important Features (Logistic Regression):")
        print(feature_importance_df.head(40))
        plt.figure(figsize=(12, 6))
        plt.barh(
            feature_importance_df["Feature"].values[:40],
            feature_importance_df["Coefficient"].values[:40],
            color="royalblue",
        )
        plt.xlabel("Coefficient")
        plt.ylabel("Feature (Word)")
        plt.title("Top 40 Most Important Features (Logistic Regression)")
        plt.gca().invert_yaxis()
        plt.show()

    return pipeline_logistic, vectorizer


def retrain_logistic_regression_model(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    solver: str = "liblinear",  # Added solver parameter
    C: float = 1.0,  # Added C parameter
    model_filename: str = "logistic_regression_model_pipeline.joblib",
) -> Pipeline:
    """
    Retrains a Logistic Regression model on the entire dataset and saves it.

    Args:
        df (pd.DataFrame): The input DataFrame containing email text and labels.
        vectorizer (TfidfVectorizer): The fitted TfidfVectorizer from the training phase.
        solver (str, optional): Solver to use for Logistic Regression. Defaults to 'liblinear'.
        C (float, optional): Regularization parameter. Defaults to 1.0
        model_filename (str, optional): Filename to save the retrained Logistic Regression model.
            Defaults to "logistic_regression_model.joblib".

    Returns:
        LogisticRegression: The retrained Logistic Regression model.
    """
    # 1. Data Loading
    df = df[["p_label", "sterilized_text"]]

    df.rename(columns={'p_label': 'Email Type', 'sterilized_text': 'Email Text'}, inplace=True)

    df['Email Text'] = df['Email Text'].fillna('')

    # 2. Feature Extraction
    X = df["Email Text"]  # Use the fitted vectorizer
    y = df["Email Type"]

    # 3. Model Training
    # Logistic Regression with scaling
    model = LogisticRegression(solver=solver, random_state=42, C=C)
    pipeline_logistic = Pipeline([
        ("tfidf", vectorizer),
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", model)
    ])
    pipeline_logistic.fit(X, y)

    # 4. Save the retrained model
    dump(pipeline_logistic, model_filename)
    return pipeline_logistic



if __name__ == "__main__":
    train_data_path = "../data/train/train_dataset.csv"
    test_data_path = "../data/test/test_dataset.csv"

    # Load the dataset
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Train the model
    trained_model, fitted_vectorizer = train_logistic_regression_model(train_data_path, test_data_path)

    # Retrain the model on the full dataset
    #final_model = retrain_logistic_regression_model(df, fitted_vectorizer)

    #print("\nFinal model retrained and saved.")
