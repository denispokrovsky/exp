import streamlit as st
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, MarianMTModel, MarianTokenizer
import matplotlib.pyplot as plt
from pymystem3 import Mystem

# Initialize pymystem3 for lemmatization
mystem = Mystem()

# Set up the sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
finbert_tone = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Function for lemmatizing Russian text
def lemmatize_text(text):
    lemmatized_text = ''.join(mystem.lemmatize(text))
    return lemmatized_text

# Translation model for Russian to English
model_name = "Helsinki-NLP/opus-mt-ru-en"
translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    inputs = translation_tokenizer(text, return_tensors="pt", truncation=True)
    translated_tokens = translation_model.generate(**inputs)
    return translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Function for VADER sentiment analysis with label mapping
def get_vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    return "Neutral"

# Functions for FinBERT, RoBERTa, and FinBERT-Tone with label mapping
def get_mapped_sentiment(result):
    label = result['label'].lower()
    if label in ["positive", "label_2", "pos", "pos_label"]:
        return "Positive"
    elif label in ["negative", "label_0", "neg", "neg_label"]:
        return "Negative"
    return "Neutral"

def get_finbert_sentiment(text):
    result = finbert(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

def get_roberta_sentiment(text):
    result = roberta(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

def get_finbert_tone_sentiment(text):
    result = finbert_tone(text, truncation=True, max_length=512)[0]
    return get_mapped_sentiment(result)

# Streamlit app setup
st.title("... ну-с, приступим...")

# File uploader
uploaded_file = st.file_uploader("грузи!", type="xlsx")
if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, sheet_name='Публикации')
    st.write("Data Preview", df.head())

    # Preprocess and translate texts
    translated_texts = []
    for i, text in enumerate(df['Выдержки из текста']):
        lemmatized_text = text #lemmatize_text(text)
        translated_text = translate(lemmatized_text)
        translated_texts.append(translated_text)
        st.write(f"Translation Progress: {((i+1)/len(df))*100:.2f}%")
        st.write (translated_text)

    # Show first five translated texts
    df_translated = df.copy()
    df_translated['Translated Text'] = translated_texts
    st.write("Translated Text Preview", df_translated[['Объект', 'Выдержки из текста', 'Translated Text']].head())

    # Button to continue to sentiment analysis
    if st.button("Continue to Sentiment Analysis"):
        # Placeholder for results
        vader_results = []
        finbert_results = []
        roberta_results = []
        finbert_tone_results = []

        # Progress indicators for each sentiment analysis method
        def process_with_progress_bar(func, name, texts):
            start_time = time.time()
            results = []
            for i, text in enumerate(texts):
                result = func(text)
                results.append(result)
                # Update progress
                elapsed_time = time.time() - start_time
                progress = (i + 1) / len(texts)
                remaining_time = elapsed_time / progress - elapsed_time
                st.write(f"{name} Progress: {progress*100:.2f}% - Estimated remaining time: {remaining_time:.2f} seconds")
            return results

        # Process each method
        with st.spinner("Processing VADER..."):
            vader_results = process_with_progress_bar(get_vader_sentiment, "VADER", translated_texts)

        with st.spinner("Processing FinBERT..."):
            finbert_results = process_with_progress_bar(get_finbert_sentiment, "FinBERT", translated_texts)

        with st.spinner("Processing RoBERTa..."):
            roberta_results = process_with_progress_bar(get_roberta_sentiment, "RoBERTa", translated_texts)

        with st.spinner("Processing FinBERT-Tone..."):
            finbert_tone_results = process_with_progress_bar(get_finbert_tone_sentiment, "FinBERT-Tone", translated_texts)

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
