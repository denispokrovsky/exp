import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

# Set up the sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Define batch size for processing
BATCH_SIZE = 100

# Function for VADER sentiment analysis with NaN handling and label mapping
def get_vader_sentiment(text):
    if isinstance(text, str):
        score = vader_analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    return None

# Functions for FinBERT, RoBERTa, and FinBERT-Tone sentiment analysis with label mapping
def get_mapped_sentiment(result):
    label = result['label']
    if label in ["POSITIVE", "LABEL_2"]:
        return "Positive"
    elif label in ["NEGATIVE", "LABEL_0"]:
        return "Negative"
    else:
        return "Neutral"

def get_finbert_sentiment(text):
    if isinstance(text, str):
        result = finbert(text, truncation=True, max_length=512)[0]
        return get_mapped_sentiment(result)
    return None

def get_roberta_sentiment(text):
    if isinstance(text, str):
        result = roberta(text, truncation=True, max_length=512)[0]
        return get_mapped_sentiment(result)
    return None

def get_finbert_tone_sentiment(text):
    if isinstance(text, str):
        result = finbert_tone(text, truncation=True, max_length=512)[0]
        return get_mapped_sentiment(result)
    return None

# Streamlit app setup
st.title("... плюс несколько методов ...")

# File uploader
uploaded_file = st.file_uploader("Загружаем и выгружаем", type="xlsx")
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, sheet_name='Публикации')
    st.write("Предпросмотр загруженного", df.head())

    # Initialize progress bars
    overall_progress = st.progress(0)
    total_steps = len(df)
    current_step = 0

    # Placeholder for results
    vader_results = []
    finbert_results = []
    roberta_results = []
    finbert_tone_results = []

    # Process data in batches
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        # Process VADER with individual progress bar
        with st.spinner("Processing VADER..."):
            vader_progress = st.progress(0)
            batch_vader = batch['Выдержки из текста'].apply(get_vader_sentiment)
            vader_results.extend(batch_vader)
            vader_progress.progress(1.0)

        # Process FinBERT with individual progress bar
        with st.spinner("Processing FinBERT..."):
            finbert_progress = st.progress(0)
            batch_finbert = batch['Выдержки из текста'].apply(get_finbert_sentiment)
            finbert_results.extend(batch_finbert)
            finbert_progress.progress(1.0)

        # Process RoBERTa with individual progress bar
        with st.spinner("Processing RoBERTa..."):
            roberta_progress = st.progress(0)
            batch_roberta = batch['Выдержки из текста'].apply(get_roberta_sentiment)
            roberta_results.extend(batch_roberta)
            roberta_progress.progress(1.0)

        # Process FinBERT-Tone with individual progress bar
        with st.spinner("Processing FinBERT-Tone..."):
            finbert_tone_progress = st.progress(0)
            batch_finbert_tone = batch['Выдержки из текста'].apply(get_finbert_tone_sentiment)
            finbert_tone_results.extend(batch_finbert_tone)
            finbert_tone_progress.progress(1.0)

        # Update overall progress bar
        current_step += len(batch)
        overall_progress.progress(min(current_step / total_steps, 1.0))

    # Add results to DataFrame
    df['VADER'] = vader_results
    df['FinBERT'] = finbert_results
    df['RoBERTa'] = roberta_results
    df['FinBERT-Tone'] = finbert_tone_results

    # Reorder columns
    columns_order = ['Объект', 'VADER', 'FinBERT', 'RoBERTa', 'FinBERT-Tone', 'Выдержки из текста']
    df = df[columns_order]
    
    # Display results
    st.write("Sentiment Analysis Results", df.head())

    # Save the output file
    output_file = "sentiment_analysis_output.xlsx"
    df.to_excel(output_file, index=False)
    
    # Download button
    with open(output_file, "rb") as file:
        st.download_button(
            label="Download output file",
            data=file,
            file_name=output_file
        )

    # Plot sentiment distribution for each model
    st.write("Sentiment Distribution")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Sentiment Distribution for Each Model")
    
    models = ['VADER', 'FinBERT', 'RoBERTa', 'FinBERT-Tone']
    for i, model in enumerate(models):
        ax = axs[i // 2, i % 2]
        sentiment_counts = df[model].value_counts()
        sentiment_counts.plot(kind='bar', ax=ax)
        ax.set_title(f"{model} Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    st.pyplot(fig)
