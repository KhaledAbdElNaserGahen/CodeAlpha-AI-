import spacy
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
from langdetect import detect
from googletrans import Translator
import pandas as pd
import re
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from collections import Counter
from streamlit_autorefresh import st_autorefresh

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
translator = Translator()

# Load FAQ dataset from Kaggle
faq_data = pd.read_csv('covid_faq.csv')

# Preprocess text using spaCy
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub('\s+', ' ', text).strip()  # Remove extra whitespaces
    text = text.lower()  # Convert to lowercase
    doc = nlp(text)  # Tokenize using spaCy
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])  # Lemmatize and remove stop words and punctuation

# Preprocess FAQ data
faq_data['question_processed'] = faq_data['questions'].apply(preprocess_text)

# Load BERT-based sentence transformer model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sentiment Analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# GPT-2 model for generating responses for unanswered questions
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# User authentication (simplified for demo purposes)
def user_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        user_id = st.text_input("Enter your user ID:")
        password = st.text_input("Enter your password:", type="password")
        if st.button("Login"):
            # Simplified check: In a real application, you'd verify credentials from a database
            if user_id == "admin" and password == "password":
                st.session_state.authenticated = True
                st.session_state.user_id = user_id
                st.success("Logged in successfully")
            else:
                st.error("Invalid credentials")
        return False
    return True

# Get best match for user query using BERT-based semantic similarity
def get_best_match(user_query):
    user_query_embeddings = sentence_model.encode([user_query])
    faq_embeddings = sentence_model.encode(faq_data['question_processed'].tolist())
    similarities = cosine_similarity(user_query_embeddings, faq_embeddings)
    best_match_idx = np.argmax(similarities)
    return faq_data.iloc[best_match_idx]['answer'], np.max(similarities)

# Get top matches for user query
def get_top_matches(user_query, top_n=3):
    user_query_embeddings = sentence_model.encode([user_query])
    faq_embeddings = sentence_model.encode(faq_data['question_processed'].tolist())
    similarities = cosine_similarity(user_query_embeddings, faq_embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_matches = [(faq_data.iloc[i]['answer'], similarities[i]) for i in top_indices]
    return top_matches

# Generate response for unanswered questions using GPT-2
def generate_response_gpt2(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=150, num_return_sequences=1)
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Named Entity Recognition (NER) Highlighting
def highlight_entities(text):
    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text, f"**{ent.text}** ({ent.label_})")
    return text

# Sentiment Analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Clustering similar questions
def cluster_questions():
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    faq_embeddings = sentence_model.encode(faq_data['question_processed'].tolist())
    X_reduced = PCA(n_components=2).fit_transform(faq_embeddings)
    dbscan.fit(X_reduced)
    return dbscan.labels_

# Keyword Extraction
def extract_keywords(text, n_keywords=5):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    return [keyword for keyword, _ in Counter(nouns).most_common(n_keywords)]

# Topic Detection
def detect_topic(text):
    doc = nlp(text)
    topics = [ent.text for ent in doc.ents if ent.label_ == 'ORG' or ent.label_ == 'GPE' or ent.label_ == 'PERSON']
    return topics

# User Profile Management
def load_user_profile(user_id):
    if os.path.exists(f"profiles/{user_id}.json"):
        with open(f"profiles/{user_id}.json", "r") as file:
            return json.load(file)
    return {}

def save_user_profile(user_id, profile):
    if not os.path.exists("profiles"):
        os.makedirs("profiles")
    with open(f"profiles/{user_id}.json", "w") as file:
        json.dump(profile, file)

# Error Handling
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    return wrapper

# Log interaction
def log_interaction(user_input, bot_response):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    with open("logs/interactions.log", "a") as log_file:
        log_file.write(json.dumps({"user_input": user_input, "bot_response": bot_response}) + "\n")

# Detect language
def detect_language(text):
    return detect(text)

# Translate text
def translate_text(text, src='auto', dest='en'):
    return translator.translate(text, src=src, dest=dest).text

# Streamlit interface
st.title("Advanced FAQ Chatbot")
st.write("Ask a question:")

@handle_errors
def main():
    if not user_authentication():
        return

    user_id = st.session_state.user_id
    user_profile = load_user_profile(user_id)

    user_input = st.text_input("You:")

    if user_input:
        detected_language = detect_language(user_input)
        if detected_language != 'en':
            user_input_translated = translate_text(user_input, src=detected_language, dest='en')
            st.write(f"Translated input: {user_input_translated}")
            user_input = user_input_translated

        top_matches = get_top_matches(user_input)
        st.write("Top matches:")
        for match, confidence in top_matches:
            st.write(f"Confidence: {confidence:.2f} - {highlight_entities(match)}")

        bot_response, confidence = get_best_match(user_input)

        if confidence < 0.5:  # Threshold to determine if the FAQ dataset has a good answer
            bot_response = generate_response_gpt2(user_input)

        bot_response = highlight_entities(bot_response)

        # Sentiment Analysis
        sentiment_label, sentiment_score = analyze_sentiment(bot_response)
        st.write(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")

        # Topic Detection
        topics = detect_topic(user_input)
        st.write(f"Detected topics: {topics}")

        # Clustering similar questions
        question_clusters = cluster_questions()

        # Keyword Extraction
        keywords = extract_keywords(user_input)
        st.write(f"Keywords: {keywords}")

        # Update user profile with the latest interaction
        user_profile["last_interaction"] = {
            "user_input": user_input,
            "bot_response": bot_response,
            "confidence": confidence,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "topics": topics,
            "question_clusters": question_clusters,
            "keywords": keywords
        }
        save_user_profile(user_id, user_profile)

        # Store context
        if 'context' not in st.session_state:
            st.session_state.context = []

        st.session_state.context.append({
            "user": user_input,
            "bot": bot_response,
            "confidence": confidence,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "topics": topics,
            "question_clusters": question_clusters,
            "keywords": keywords
        })

        # Log interaction
        log_interaction(user_input, bot_response)

        st.write(f"Bot (Confidence: {confidence:.2f}):", bot_response)

        # Feedback mechanism
        feedback = st.radio("Was this answer helpful?", ("Yes", "No"))
        if feedback:
            with open("logs/feedback.log", "a") as feedback_file:
                feedback_file.write(json.dumps({"user_input": user_input, "bot_response": bot_response, "feedback": feedback}) + "\n")
            st.write("Thank you for your feedback!")

        # Real-time notifications
        if st_autorefresh(interval=60000, limit=100, key="notif"):
            st.write("Checking for updates...")

    # Display context
    if 'context' in st.session_state and st.session_state.context:
        st.write("Conversation Context:")
        for entry in st.session_state.context:
            st.write(f"You: {entry['user']}")
            st.write(f"Bot: {entry['bot']}")
            st.write(f"Sentiment: {entry['sentiment_label']} (Score: {entry['sentiment_score']:.2f})")
            st.write(f"Topics: {entry['topics']}")
            st.write(f"Question Clusters: {entry['question_clusters']}")
            st.write(f"Keywords: {entry['keywords']}")

    # Graphical Data Visualization (e.g., sentiment trends)
    st.write("Sentiment Trends")
    if 'context' in st.session_state and st.session_state.context:
        sentiment_scores = [entry['sentiment_score'] for entry in st.session_state.context]
        st.line_chart(sentiment_scores)

if __name__ == "__main__":
    main()
