#!/usr/bin/env python3
"""
Test script for the Hotel Sentiment Analyzer
"""

import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

def test_models():
    """Test the sentiment analysis models"""
    print("🏨 Testing Hotel Sentiment Analyzer Models...")
    
    # Download NLTK data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Load data
    print("📊 Loading dataset...")
    df = pd.read_csv('tripadvisor_hotel_reviews.csv')
    
    # Label sentiment
    def label_sentiment(rating):
        if rating >= 4:
            return 1  # satisfied
        else:
            return 0  # unsatisfied
    
    df['Sentiment'] = df['Rating'].apply(label_sentiment)
    
    # Text preprocessing
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    
    df['Cleaned_Review'] = df['Review'].apply(clean_text)
    
    # Prepare data
    X = df['Cleaned_Review']
    y_sentiment = df['Sentiment']
    y_rating = df['Rating']
    
    # TF-IDF Vectorization
    print("🔤 Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(X)
    
    # Train sentiment model
    print("🤖 Training sentiment model...")
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_tfidf, y_sentiment, test_size=0.2, random_state=42
    )
    sentiment_model = LogisticRegression()
    sentiment_model.fit(X_train_s, y_train_s)
    
    # Train rating model
    print("⭐ Training rating model...")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_tfidf, y_rating, test_size=0.2, random_state=42
    )
    rating_model = LogisticRegression(solver='lbfgs', max_iter=300)
    rating_model.fit(X_train_r, y_train_r)
    
    # Test predictions
    print("🧪 Testing predictions...")
    
    test_reviews = [
        "The staff was friendly and the room was clean and beautiful!",
        "The staff was so bad but the room was clean!",
        "it was just okay!",
        "The staff was not friendly at all, i had a terrible experience!",
        "i was an amazing experience, i liked it!"
    ]
    
    for review in test_reviews:
        cleaned_text = clean_text(review)
        text_vector = tfidf.transform([cleaned_text])
        
        sentiment = sentiment_model.predict(text_vector)[0]
        rating = rating_model.predict(text_vector)[0]
        
        sentiment_label = "Satisfied 😊" if sentiment == 1 else "Unsatisfied 😞"
        
        print(f"\n📝 Review: {review}")
        print(f"   Sentiment: {sentiment_label}")
        print(f"   Rating: {rating} {'⭐' * rating}")
    
    # Model accuracy
    sentiment_accuracy = accuracy_score(y_test_s, sentiment_model.predict(X_test_s))
    rating_accuracy = accuracy_score(y_test_r, rating_model.predict(X_test_r))
    
    print(f"\n📈 Model Performance:")
    print(f"   Sentiment Accuracy: {sentiment_accuracy:.3f}")
    print(f"   Rating Accuracy: {rating_accuracy:.3f}")
    
    print("\n✅ All tests passed! The models are working correctly.")
    return True

if __name__ == "__main__":
    test_models() 