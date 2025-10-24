# app.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import StringIO, BytesIO
import time
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import joblib
import os
import json
from datetime import datetime

# ---------------------------
# NLTK initialization w/ error handling
# ---------------------------
def initialize_nltk():
    try:
        nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path)
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.append(nltk_data_path)

        required_nltk = [
            ('punkt', 'tokenizers/punkt'),
            ('wordnet', 'corpora/wordnet'),
            ('stopwords', 'corpora/stopwords'),
            ('omw-1.4', 'corpora/omw-1.4'),
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
        ]

        for resource, path in required_nltk:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)
    except Exception as e:
        st.error(f"Failed to initialize NLTK: {str(e)}")
        st.stop()

# initialize
initialize_nltk()

# Deterministic language detection
DetectorFactory.seed = 0

# Initialize NLP components
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    vader_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    st.error(f"Failed to initialize NLP components: {str(e)}")
    st.stop()

# Streamlit page config
st.set_page_config(
    page_title="Election Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS (unchanged, but included) ----------
st.markdown("""
<style>
    :root {
        --primary-dark: #1a237e;
        --primary-light: #3949ab;
        --secondary: #00acc1;
        --accent: #ff7043;
        --text-primary: #e8eaf6;
        --text-secondary: #c5cae9;
        --success: #4caf50;
        --danger: #f44336;
        --warning: #ff9800;
        --info: #2196f3;
    }
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; transition: all 0.3s ease; }
    .header { background: linear-gradient(135deg, var(--primary-dark), var(--primary-light)); color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); margin-bottom: 2rem; text-align: center; border-bottom: 4px solid var(--secondary); }
    .party-card { background: white; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 4px solid var(--secondary); transition: all 0.3s ease; color: black; }
    .result-card { background: white; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); animation: slideUp 0.5s ease-out; color: black; }
    .winner-card { background: linear-gradient(135deg, var(--success), #66bb6a); color: white ; border-left: 6px solid white; animation: pulse 2s infinite; }
    .positive { color: var(--success); font-weight: 600; }
    .negative { color: var(--danger); font-weight: 600; }
    .neutral { color: var(--warning); font-weight: 600; }
    .footer { background: linear-gradient(135deg, var(--primary-dark), var(--primary-light)); color: white; padding: 2rem; border-radius: 10px; margin-top: 3rem; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, var(--secondary), var(--accent)); }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1 style="margin:0;font-size:2.5rem;">📊 Election Sentiment Analysis Dashboard</h1>
    <p style="margin:0.5rem 0 0;font-size:1.1rem;">Advanced political sentiment analysis with hybrid NLP approaches</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Utility functions
# ---------------------------
@lru_cache(maxsize=10000)
def detect_language_cached(text):
    if not text or not isinstance(text, str):
        return 'en'
    try:
        return detect(text)
    except Exception:
        return 'en'

def safe_tokenize(text):
    try:
        return word_tokenize(text)
    except Exception:
        # fallback: simple split
        return text.split()

def advanced_preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert emojis to text
    text = emoji.demojize(text)
    # Remove URLs, mentions, hashtags (keep the word after #)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'\n', ' ', text)
    # Expand contractions (simple)
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    # Tokenize and lemmatize
    try:
        tokens = safe_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        return ' '.join(tokens)
    except Exception:
        return text.lower().strip()

@lru_cache(maxsize=10000)
def cached_sentiment_analysis(text, method):
    # caching by text and method
    if not text:
        return 0.0
    try:
        if method == 'textblob':
            return TextBlob(text).sentiment.polarity
        elif method == 'vader':
            return vader_analyzer.polarity_scores(text)['compound']
        else:
            return (TextBlob(text).sentiment.polarity + vader_analyzer.polarity_scores(text)['compound']) / 2
    except Exception:
        return 0.0

def categorize_sentiment(score):
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

def process_tweet_batch(tweets, analysis_method):
    processed_tweets = []
    sentiments = []
    scores = []
    for tweet in tweets:
        try:
            processed_text = advanced_preprocess_text(tweet)
            score = cached_sentiment_analysis(processed_text, analysis_method)
            sentiment = categorize_sentiment(score)
            processed_tweets.append(processed_text)
            sentiments.append(sentiment)
            scores.append(score)
        except Exception:
            # continue on error, returning best-effort
            processed_tweets.append("")
            sentiments.append("neutral")
            scores.append(0.0)
    return processed_tweets, sentiments, scores

# Visualization helpers
def create_sentiment_distribution_plot(party_name, positive, negative, neutral):
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive, negative, neutral]
    colors = ['#4CAF50', '#F44336', '#FFC107']
    explode = (0.05, 0.05, 0.05)
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, shadow=True)
    ax.set_title(f"Sentiment Distribution for {party_name}", fontsize=14)
    ax.axis('equal')
    plt.tight_layout()
    return fig

def create_sentiment_trend_plot(df, party_name):
    fig, ax = plt.subplots(figsize=(10, 4))
    try:
        weekly_sentiment = df.resample('W', on='date')['sentiment_score'].mean()
        weekly_sentiment.plot(ax=ax, linewidth=2.2, marker='o')
        ax.axhline(0.05, color='#4CAF50', linestyle='--', alpha=0.7, label='Positive Threshold')
        ax.axhline(-0.05, color='#F44336', linestyle='--', alpha=0.7, label='Negative Threshold')
        ax.set_title(f'Sentiment Trend Over Time - {party_name}', pad=10)
        ax.set_ylabel('Average Sentiment Score')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception:
        return None

def create_comparison_chart(results):
    fig, ax = plt.subplots(figsize=(10, 4))
    parties = [r['party_name'] for r in results]
    positives = [r['positive'] for r in results]
    negatives = [r['negative'] for r in results]
    neutrals = [r['neutral'] for r in results]
    x = np.arange(len(parties))
    width = 0.25
    bar1 = ax.bar(x - width, positives, width, label='Positive')
    bar2 = ax.bar(x, negatives, width, label='Negative')
    bar3 = ax.bar(x + width, neutrals, width, label='Neutral')
    ax.set_xticks(x)
    ax.set_xticklabels(parties)
    ax.set_ylabel('Percentage')
    ax.set_title('Sentiment Comparison Across Parties')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    return fig

# ML training (cached to speed re-runs)
@st.cache_data(show_spinner=False)
def train_ml_model(tweets, labels):
    # Defensive: need enough data
    if len(tweets) < 10 or len(set(labels)) < 2:
        return None, None
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(tweets)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)
    models = {
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            conf = confusion_matrix(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': conf
            }
        except Exception as e:
            results[name] = {
                'model': None,
                'accuracy': 0.0,
                'report': {},
                'confusion_matrix': None,
                'error': str(e)
            }
    return vectorizer, results

# Analyze party
@st.cache_data(show_spinner=False)
def analyze_party_sentiment(uploaded_file, party_name, analysis_method='hybrid', max_tweets=2000):
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
    except Exception as e:
        # If file couldn't be read
        return {'error': f"Could not read CSV for {party_name}: {str(e)}"}

    if 'text' not in df.columns:
        return {'error': f"CSV for {party_name} must contain a 'text' column."}

    # limit sampling
    if len(df) > max_tweets:
        df = df.sample(max_tweets, random_state=42).reset_index(drop=True)

    tweets = df['text'].astype(str).tolist()
    batch_size = 500
    batches = [tweets[i:i+batch_size] for i in range(0, len(tweets), batch_size)]
    # Use threadpool to parallelize text processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda batch: process_tweet_batch(batch, analysis_method), batches))

    processed_tweets = []
    sentiments = []
    scores = []
    for batch_result in results:
        processed_tweets.extend(batch_result[0])
        sentiments.extend(batch_result[1])
        scores.extend(batch_result[2])

    total = max(1, len(sentiments))
    positive_pct = sentiments.count('positive') / total * 100
    negative_pct = sentiments.count('negative') / total * 100
    neutral_pct = sentiments.count('neutral') / total * 100

    dist_fig = create_sentiment_distribution_plot(party_name, positive_pct, negative_pct, neutral_pct)
    trend_fig = None
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['sentiment_score'] = scores
            df['sentiment'] = sentiments
            # drop rows w/o valid date
            if df['date'].notnull().sum() > 0:
                trend_fig = create_sentiment_trend_plot(df.dropna(subset=['date']), party_name)
        except Exception:
            trend_fig = None

    # prepare processed dataframe for download / display
    processed_df = pd.DataFrame({
        'original_text': df['text'].astype(str),
        'processed_text': processed_tweets[:len(df)],
        'sentiment': sentiments[:len(df)],
        'sentiment_score': scores[:len(df)]
    })

    return {
        'party_name': party_name,
        'positive': positive_pct,
        'negative': negative_pct,
        'neutral': neutral_pct,
        'sentiment_distribution': dist_fig,
        'sentiment_trend': trend_fig,
        'processed_tweets': processed_tweets,
        'sentiments': sentiments,
        'scores': scores,
        'raw_data': df,
        'processed_df': processed_df
    }

# ---------------------------
# Main UI and orchestration
# ---------------------------
def main():
    # session state
    if 'party_count' not in st.session_state:
        st.session_state.party_count = 2
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'ml_vectorizer' not in st.session_state:
        st.session_state.ml_vectorizer = None

    # sidebar settings
    st.sidebar.markdown("### Analysis Configuration")
    with st.sidebar.expander("Analysis Settings", expanded=True):
        analysis_method = st.selectbox(
            "Sentiment Analysis Method",
            ["Hybrid (TextBlob + VADER)", "TextBlob Only", "VADER Only"],
            index=0
        )
        enable_ml = st.checkbox("Enable ML Comparison", value=True)
        max_tweets = st.slider("Maximum tweets per party", min_value=100, max_value=5000, value=2000, step=100)

    st.markdown("<div style='text-align:center;margin-bottom:1rem;'><h2>Upload Party Data</h2><p>CSV files should contain at least a 'text' column</p></div>", unsafe_allow_html=True)

    cols_per_row = 3
    party_files = {}
    rows = (st.session_state.party_count + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            party_idx = row * cols_per_row + col_idx
            if party_idx < st.session_state.party_count:
                with cols[col_idx]:
                    st.markdown(f"<div class='party-card'><h3>Party {party_idx + 1}</h3></div>", unsafe_allow_html=True)
                    party_name = st.text_input("Enter party name", key=f"party_name_{party_idx}", value=f"Party {party_idx + 1}", label_visibility="collapsed")
                    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key=f"uploader_{party_idx}", label_visibility="collapsed")
                    if uploaded_file:
                        party_files[party_name] = uploaded_file

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("＋ Add Party", key="add_party"):
            st.session_state.party_count += 1
            st.experimental_rerun()
    with col2:
        analyze_clicked = st.button("🚀 Analyze Sentiment", key="analyze_btn", type="primary")

    # Run analysis
    if analyze_clicked:
        # count valid uploaded files
        valid_parties = sum(1 for i in range(st.session_state.party_count) if st.session_state.get(f"uploader_{i}") is not None)
        if valid_parties < 2:
            st.error("Please upload CSV files for at least two parties.")
            st.stop()

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        status_text.text('Starting analysis...')
        st.session_state.results = []
        st.session_state.analyzed = True

        method_map = {
            "Hybrid (TextBlob + VADER)": "hybrid",
            "TextBlob Only": "textblob",
            "VADER Only": "vader"
        }

        all_processed_tweets = []
        all_labels = []
        i = 0
        for idx in range(st.session_state.party_count):
            party_name = st.session_state.get(f"party_name_{idx}", f"Party {idx + 1}")
            uploaded_file = st.session_state.get(f"uploader_{idx}")
            if uploaded_file:
                status_text.text(f'Analyzing {party_name}...')
                result = analyze_party_sentiment(uploaded_file, party_name, method_map[analysis_method], max_tweets)
                if result and 'error' not in result:
                    st.session_state.results.append(result)
                    # extend global lists for ML training
                    processed = result['processed_tweets'][:len(result['raw_data'])]
                    labels = result['sentiments'][:len(result['raw_data'])]
                    all_processed_tweets.extend(processed)
                    all_labels.extend(labels)
                else:
                    st.warning(f"Skipping {party_name} due to error: {result.get('error') if result else 'Unknown error'}")

                i += 1
                progress_bar.progress(min(i / st.session_state.party_count, 1.0))
                time.sleep(0.2)

        status_text.text('Analysis complete.')

        # Display party results
        st.markdown("## Party Results")
        results_cols = st.columns(len(st.session_state.results))
        for idx, res in enumerate(st.session_state.results):
            with results_cols[idx]:
                st.pyplot(res['sentiment_distribution'])
                st.markdown(f"**{res['party_name']}**")
                st.markdown(f"- Positive: <span class='positive'>{res['positive']:.2f}%</span>", unsafe_allow_html=True)
                st.markdown(f"- Negative: <span class='negative'>{res['negative']:.2f}%</span>", unsafe_allow_html=True)
                st.markdown(f"- Neutral: <span class='neutral'>{res['neutral']:.2f}%</span>", unsafe_allow_html=True)
                # download processed data for this party
                csv_bytes = res['processed_df'].to_csv(index=False).encode('utf-8')
                st.download_button(label="Download processed CSV", data=csv_bytes, file_name=f"{res['party_name']}_processed.csv", mime='text/csv')

        # Comparison chart
        if len(st.session_state.results) >= 2:
            st.markdown("## Comparison across parties")
            comp_fig = create_comparison_chart(st.session_state.results)
            st.pyplot(comp_fig)

            # Determine winner by highest positive %
            winner = max(st.session_state.results, key=lambda x: x['positive'])
            st.markdown(f"<div class='winner-card result-card'><h3 style='margin:0'>{winner['party_name']} — Most Positive Sentiment</h3><p style='margin:0.25rem 0 0'>Positive: {winner['positive']:.2f}%</p></div>", unsafe_allow_html=True)

        # Show time trend if available
        st.markdown("## Time Trends (if available)")
        for res in st.session_state.results:
            if res['sentiment_trend'] is not None:
                st.markdown(f"### {res['party_name']}")
                st.pyplot(res['sentiment_trend'])

        # Top positive/negative tweets panels
        st.markdown("## Top Tweets")
        for res in st.session_state.results:
            st.markdown(f"### {res['party_name']}")
            df_proc = res['processed_df']
            if 'sentiment_score' not in df_proc.columns:
                df_proc['sentiment_score'] = res.get('scores', [0]*len(df_proc))[:len(df_proc)]
            top_pos = df_proc.sort_values('sentiment_score', ascending=False).head(5)
            top_neg = df_proc.sort_values('sentiment_score').head(5)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Top Positive**")
                for _, row in top_pos.iterrows():
                    st.markdown(f"> {row['original_text'][:300]}  ")
                    st.caption(f"Score: {row['sentiment_score']:.3f}")
            with cols[1]:
                st.markdown("**Top Negative**")
                for _, row in top_neg.iterrows():
                    st.markdown(f"> {row['original_text'][:300]}  ")
                    st.caption(f"Score: {row['sentiment_score']:.3f}")

        # Train ML models if enabled
        if enable_ml and len(all_processed_tweets) > 100 and len(set(all_labels)) > 1:
            status_text.text('Training machine learning models...')
            try:
                vectorizer, ml_results = train_ml_model(all_processed_tweets, all_labels)
                if vectorizer is None or ml_results is None:
                    st.warning("Insufficient/invalid data for ML training.")
                else:
                    st.session_state.ml_vectorizer = vectorizer
                    st.session_state.ml_results = ml_results

                    st.markdown("## ML Model Comparison")
                    # build a results table
                    rows = []
                    for name, info in ml_results.items():
                        rows.append({
                            'Model': name,
                            'Accuracy': info.get('accuracy', 0.0)
                        })
                    ml_df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False)
                    st.dataframe(ml_df)

                    # display detailed reports
                    for name, info in ml_results.items():
                        st.markdown(f"### {name}")
                        if info.get('model') is None:
                            st.warning(f"{name}: training failed. Error: {info.get('error')}")
                            continue
                        st.write(f"Accuracy: {info.get('accuracy', 0.0):.3f}")
                        report = info.get('report', {})
                        if report:
                            # pretty print per-class metrics
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                        # confusion matrix
                        conf = info.get('confusion_matrix')
                        if conf is not None:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            sns.heatmap(conf, annot=True, fmt='d', ax=ax)
                            ax.set_title(f"{name} Confusion Matrix")
                            plt.tight_layout()
                            st.pyplot(fig)

                    # persist best model + vectorizer
                    best_model_name = max(ml_results.items(), key=lambda x: x[1].get('accuracy', 0))[0]
                    best_model = ml_results[best_model_name]['model']
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_dir = os.path.join(os.getcwd(), "saved_models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"best_model_{best_model_name.replace(' ','_')}_{timestamp}.joblib")
                    vec_path = os.path.join(model_dir, f"vectorizer_{timestamp}.joblib")
                    joblib.dump(best_model, model_path)
                    joblib.dump(vectorizer, vec_path)

                    # download buttons
                    with st.expander("Download trained artifacts"):
                        with open(model_path, "rb") as f:
                            st.download_button("Download Best Model (.joblib)", f, file_name=os.path.basename(model_path))
                        with open(vec_path, "rb") as f:
                            st.download_button("Download Vectorizer (.joblib)", f, file_name=os.path.basename(vec_path))

            except Exception as e:
                st.error(f"Error training ML models: {str(e)}")
            finally:
                status_text.text('Complete.')
        else:
            if enable_ml:
                st.info("Not enough data or not enough label diversity to train ML models. Need >100 samples and at least 2 classes.")
            st.session_state.ml_results = None

        # Combined processed CSV download (all parties)
        if len(st.session_state.results) > 0:
            combined = []
            for res in st.session_state.results:
                dfp = res['processed_df'].copy()
                dfp['party_name'] = res['party_name']
                combined.append(dfp)
            combined_df = pd.concat(combined, ignore_index=True)
            csv_bytes = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download combined processed data (CSV)", csv_bytes, file_name="combined_processed_tweets.csv", mime="text/csv")

    # Footer
    st.markdown("<div class='footer'><div style='display:flex;justify-content:space-between;align-items:center;'><div><h3 style='margin:0'>Done ✅</h3><p style='margin:0.2rem 0 0'>Use the download buttons to fetch processed data and models.</p></div><div style='text-align:right'><small>Built with ❤️ and NLP</small></div></div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
