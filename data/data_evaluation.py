#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from tqdm.auto import tqdm
from sklearn.decomposition import TruncatedSVD

# 1) Make sure punkt models are present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def load_data(path):
    df = pd.read_csv(path)
    print("Data loaded:", df.shape)
    return df

def eda(df, output_dir):
    # Class distributions
    print("\nPhishing label distribution:\n", df['p_label'].value_counts(normalize=True))
    print("\nGenerated label distribution:\n", df['g_label'].value_counts(normalize=True))
    print("\nJoint distribution:\n", pd.crosstab(df['p_label'], df['g_label'], normalize='all'))

    # Origin breakdown
    print("\nOrigin counts:\n", df['origin'].value_counts())

    # Missing values
    print("\nMissing values:\n", df.isnull().sum())

    # Text length histogram
    lengths = df['text'].str.len()
    plt.figure()
    lengths_capped = lengths[lengths <= 2500]
    plt.hist(lengths_capped, bins=50)
    plt.title("Email Text Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
    plt.close()

def text_stats(df):
    df['char_count'] = df['text'].str.len()
    df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    df['sent_count'] = df['text'].apply(lambda x: len(sent_tokenize(x)))
    df['flesch_reading_ease'] = df['text'].apply(lambda x: textstat.flesch_reading_ease(x))
    print("\nSample text stats:\n", df[['char_count','word_count','sent_count','flesch_reading_ease']].describe())



def plot_top20_comparison(df,
                          text_col='text',
                          label_col='p_label',
                          top_n=20,
                          output_dir="data/results",
                          filename="top20_word_comparison.png"):
    """
    1) Finds the top_n words by overall normalized frequency.
    2) Computes the normalized frequency of those words in each class (0 and 1).
    3) Plots a grouped bar chart and optionally saves it to output_dir/filename.
    Returns the DataFrame of frequencies.
    """
    print("hi")
    # ---- 1) overall frequencies ----
    vect_all = CountVectorizer(stop_words='english')
    X_all = vect_all.fit_transform(df[text_col])
    counts_all = np.asarray(X_all.sum(axis=0)).ravel()
    counts_all = counts_all / counts_all.sum()
    words = np.array(vect_all.get_feature_names_out())
    top_idx = np.argsort(counts_all)[::-1][:top_n]
    top_words = words[top_idx]

    # ---- 2) class frequencies ----
    vect0 = CountVectorizer(stop_words='english', vocabulary=top_words)
    vect1 = CountVectorizer(stop_words='english', vocabulary=top_words)
    X0 = vect0.fit_transform(df.loc[df[label_col]==0, text_col])
    X1 = vect1.fit_transform(df.loc[df[label_col]==1, text_col])
    counts0 = np.asarray(X0.sum(axis=0)).ravel()
    counts1 = np.asarray(X1.sum(axis=0)).ravel()
    counts0 = counts0 / counts0.sum() if counts0.sum() > 0 else counts0
    counts1 = counts1 / counts1.sum() if counts1.sum() > 0 else counts1

    # ---- 3) assemble DataFrame ----
    df_compare = pd.DataFrame({
        'overall':    counts_all[top_idx],
        'p_label=0':  counts0,
        'p_label=1':  counts1
    }, index=top_words)

    # ---- 4) plot grouped bar chart ----
    plt.figure(figsize=(12, 6))
    df_compare.plot(kind='bar', width=0.8)
    plt.title(f"Top {top_n} words: overall vs. p_label=0 vs. p_label=1")
    plt.xlabel("Word")
    plt.ylabel("Normalized frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # ---- 5) save to file if requested ----
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.close()
    print("out")
    return df_compare





def lexical_richness(df):
    df['uniq_words'] = df['text'].apply(lambda x: len(set(word_tokenize(x.lower()))))
    df['type_token_ratio'] = df['uniq_words'] / df['word_count']
    print("\nLexical richness stats:\n", df[['uniq_words','type_token_ratio']].describe())

def url_email_patterns(df):
    df['urls'] = df['text'].apply(lambda x: re.findall(r'https?://\S+|www\.\S+', x))
    df['emails'] = df['text'].apply(lambda x: re.findall(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', x))
    print("\nAverage URLs per email:", df['urls'].apply(len).mean())
    print("Average EMAILs per email:", df['emails'].apply(len).mean())

def ngram_analysis(df, label_col, top_n=20):
    vect = CountVectorizer(stop_words='english', ngram_range=(1,2))
    X = vect.fit_transform(df['text'])
    feature_names = np.array(vect.get_feature_names_out())
    for label in sorted(df[label_col].unique()):
        mask = (df[label_col] == label).to_numpy()
        counts = X[mask].sum(axis=0).A1
        top = feature_names[np.argsort(counts)[-top_n:]]
        print(f"\nTop {top_n} n-grams for {label_col}={label}:\n", top)

def topic_modeling(df, n_topics=10):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['text'])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)
    terms = tfidf.get_feature_names_out()
    for i, comp in enumerate(lda.components_):
        top_terms = terms[np.argsort(comp)[-10:]]
        print(f"Topic {i}: {', '.join(top_terms)}")

def embeddings_tsne_fast(df, output_dir,
                         sample_size=50000,
                         tfidf_max_features=2000,
                         svd_components=50,
                         tsne_perplexity=30):
    # 1) Subsample
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=0)
    else:
        df_sample = df

    # 2) TF-IDF vectorization (reduced vocab)
    vect = TfidfVectorizer(max_features=tfidf_max_features,
                           stop_words='english')
    X = vect.fit_transform(df_sample['text'])

    # 3) Truncated SVD → ~50 dims
    svd = TruncatedSVD(n_components=svd_components, random_state=0)
    X_reduced = svd.fit_transform(X)

    # 4) t-SNE on the 50-dim data
    tsne = TSNE(n_components=2,
                init='pca',
                learning_rate='auto',
                perplexity=tsne_perplexity,
                random_state=0)
    emb = tsne.fit_transform(X_reduced)

    # 5) Plot & save
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1],
                c=df_sample['p_label'],
                alpha=0.6,
                s=5)
    plt.title("t-SNE of TF-IDF → SVD → 2D")
    plt.xlabel("TSNE‐1")
    plt.ylabel("TSNE‐2")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "tsne_tfidf_fast.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved accelerated t-SNE plot to {out_path}")

import signal

# 1. Define a custom exception for timeouts
class TimeoutException(Exception):
    pass

def timeout(seconds=1.0):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException()
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                return func(*args, **kwargs)
            except TimeoutException:
                return np.nan
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        return wrapper
    return decorator

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

@timeout(seconds=1.0)
def safe_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

def sentiment_analysis_split(df):
    """
    Compute and print sentiment stats separately for:
      - p_label == 0 and p_label == 1 (dropping NaNs)
      - g_label == 0 and g_label == 1 (dropping NaNs)
    """
    for label_type, col in [('p_label', 'p_label'), ('g_label', 'g_label')]:
        # Drop samples without this label
        df_labeled = df.dropna(subset=[col])
        for val in [0, 1]:
            subset = df_labeled[df_labeled[col] == val]
            # Compute safe sentiment
            sentiments = subset['text'].progress_apply(safe_sentiment)
            print(f"\nSentiment stats for {label_type} = {val} ({len(subset)} samples):")
            print(sentiments.describe())
    print("\nComputing overall sentiment distribution...")
    all_sentiments = df['text'].progress_apply(safe_sentiment)
    print("\nOverall sentiment summary:")
    print(all_sentiments.describe())

def stylometry(df):
    df['avg_sent_len'] = df['word_count'] / df['sent_count']
    df['punct_count'] = df['text'].progress_apply(
        lambda x: len(re.findall(r'[^\w\s]', x))
    )
    print("\nStylometry stats by machine-generated label:\n",
          df.groupby('g_label')[['avg_sent_len','punct_count']].describe())
    print("\nStylometry stats by machine-generated label:\n",
          df.groupby('p_label')[['avg_sent_len','punct_count']].describe())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Path to CSV file")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.input)
    df['text'] = df['text'].fillna('').astype(str)
    tqdm.pandas(desc="Overall")

    eda(df, args.output_dir)
    text_stats(df)
    plot_top20_comparison(df)
    lexical_richness(df)
    url_email_patterns(df)
    ngram_analysis(df, 'p_label')
    topic_modeling(df)
    embeddings_tsne_fast(df, args.output_dir)
    sentiment_analysis_split(df)
    stylometry(df)

if __name__ == "__main__":
    main()
