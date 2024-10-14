import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt

# Set up the sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Function for VADER sentiment analysis
def get_vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)["compound"]

# Function for FinBERT sentiment analysis
def get_finbert_sentiment(text):
    result = finbert(text)[0]
    return result['label']

# Function for RoBERTa sentiment analysis
def get_roberta_sentiment(text):
    result = roberta(text)[0]
    return result['label']

# Function for FinBERT-Tone sentiment analysis
def get_finbert_tone_sentiment(text):
    result = finbert_tone(text)[0]
    return result['label']

# Streamlit app setup
st.title("Financial News Sentiment Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, sheet_name='Публикации')
    st.write("Data Preview", df.head())

    # Apply sentiment analysis models
    df['VADER'] = df['Выдержки из текста'].apply(get_vader_sentiment)
    df['FinBERT'] = df['Выдержки из текста'].apply(get_finbert_sentiment)
    df['RoBERTa'] = df['Выдержки из текста'].apply(get_roberta_sentiment)
    df['FinBERT-Tone'] = df['Выдержки из текста'].apply(get_finbert_tone_sentiment)

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
