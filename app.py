import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords

# Page configuration
st.set_page_config(
    page_title="Hotel Sentiment Analyzer",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    df = pd.read_csv('tripadvisor_hotel_reviews.csv')
    return df

@st.cache_resource
def load_models():
    """Load and cache the trained models"""
    # Download NLTK data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Load data
    df = load_data()
    
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
    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(X)
    
    # Train sentiment model
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_tfidf, y_sentiment, test_size=0.2, random_state=42
    )
    sentiment_model = LogisticRegression()
    sentiment_model.fit(X_train_s, y_train_s)
    
    # Train rating model
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_tfidf, y_rating, test_size=0.2, random_state=42
    )
    rating_model = LogisticRegression(solver='lbfgs', max_iter=300)
    rating_model.fit(X_train_r, y_train_r)
    
    return tfidf, sentiment_model, rating_model, df, X_test_s, y_test_s, X_test_r, y_test_r

def predict_sentiment(text, tfidf, sentiment_model, rating_model):
    """Predict sentiment and rating for given text"""
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    
    cleaned_text = clean_text(text)
    text_vector = tfidf.transform([cleaned_text])
    
    sentiment = sentiment_model.predict(text_vector)[0]
    rating = rating_model.predict(text_vector)[0]
    
    return sentiment, rating, cleaned_text

def main():
    # Header
    st.markdown('<h1 class="main-header">🏨 Hotel Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        tfidf, sentiment_model, rating_model, df, X_test_s, y_test_s, X_test_r, y_test_r = load_models()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Analysis", "🔍 Single Review", "📁 Batch Analysis", "📈 Model Performance"]
    )
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Analysis":
        show_analysis_page(df)
    elif page == "🔍 Single Review":
        show_single_review_page(tfidf, sentiment_model, rating_model)
    elif page == "📁 Batch Analysis":
        show_batch_analysis_page(tfidf, sentiment_model, rating_model)
    elif page == "📈 Model Performance":
        show_model_performance_page(sentiment_model, rating_model, X_test_s, y_test_s, X_test_r, y_test_r)

def show_home_page(df):
    """Display the home page with overview"""
    st.markdown("## Welcome to Hotel Sentiment Analyzer! 🎉")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    
    with col2:
        avg_rating = df['Rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    
    with col3:
        satisfied_count = len(df[df['Rating'] >= 4])
        satisfaction_rate = (satisfied_count / len(df)) * 100
        st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
    
    st.markdown("---")
    
    # Quick insights
    st.markdown("### 📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        fig = px.histogram(df, x='Rating', nbins=5, 
                          title="Rating Distribution",
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment distribution
        df['Sentiment_Label'] = df['Rating'].apply(lambda x: 'Satisfied' if x >= 4 else 'Unsatisfied')
        sentiment_counts = df['Sentiment_Label'].value_counts()
        
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title="Sentiment Distribution",
                     color_discrete_sequence=['#28a745', '#dc3545'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Features
    st.markdown("### ✨ Features")
    features = [
        "🔍 **Single Review Analysis**: Analyze individual hotel reviews",
        "📁 **Batch Analysis**: Upload multiple reviews for analysis",
        "📊 **Data Visualization**: Interactive charts and insights",
        "🤖 **ML Models**: Sentiment classification and rating prediction",
        "📈 **Performance Metrics**: Model accuracy and evaluation"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")

def show_analysis_page(df):
    """Display data analysis and visualizations"""
    st.markdown("## 📊 Data Analysis & Insights")
    
    # Data overview
    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("**Dataset Statistics:**")
        st.write(f"- **Total Reviews**: {len(df):,}")
        st.write(f"- **Average Rating**: {df['Rating'].mean():.2f}")
        st.write(f"- **Rating Range**: {df['Rating'].min()} - {df['Rating'].max()}")
        st.write(f"- **Most Common Rating**: {df['Rating'].mode().iloc[0]}")
    
    st.markdown("---")
    
    # Interactive visualizations
    st.markdown("### 📈 Interactive Visualizations")
    
    # Rating distribution with Plotly
    fig = px.histogram(df, x='Rating', nbins=5, 
                      title="Hotel Rating Distribution",
                      labels={'Rating': 'Star Rating', 'count': 'Number of Reviews'},
                      color_discrete_sequence=['#1f77b4'])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis
    df['Sentiment_Label'] = df['Rating'].apply(lambda x: 'Satisfied' if x >= 4 else 'Unsatisfied')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment pie chart
        sentiment_counts = df['Sentiment_Label'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                     title="Sentiment Distribution",
                     color_discrete_sequence=['#28a745', '#dc3545'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment bar chart
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                     title="Sentiment Counts",
                     labels={'x': 'Sentiment', 'y': 'Count'},
                     color=sentiment_counts.index,
                     color_discrete_map={'Satisfied': '#28a745', 'Unsatisfied': '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Word cloud (if reviews are available)
    st.markdown("### 📝 Review Length Analysis")
    
    df['Review_Length'] = df['Review'].str.len()
    df['Word_Count'] = df['Review'].str.split().str.len()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Review_Length', nbins=50,
                          title="Review Length Distribution",
                          labels={'Review_Length': 'Character Count', 'count': 'Number of Reviews'},
                          color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Word_Count', nbins=30,
                          title="Word Count Distribution",
                          labels={'Word_Count': 'Number of Words', 'count': 'Number of Reviews'},
                          color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)

def show_single_review_page(tfidf, sentiment_model, rating_model):
    """Display single review analysis page"""
    st.markdown("## 🔍 Single Review Analysis")
    
    st.markdown("Enter a hotel review to analyze its sentiment and predict the rating:")
    
    # Text input
    review_text = st.text_area(
        "Hotel Review:",
        placeholder="Enter your hotel review here...",
        height=150
    )
    
    if st.button("Analyze Review", type="primary"):
        if review_text.strip():
            with st.spinner("Analyzing review..."):
                sentiment, rating, cleaned_text = predict_sentiment(
                    review_text, tfidf, sentiment_model, rating_model
                )
            
            # Results
            st.markdown("### 📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_label = "Satisfied 😊" if sentiment == 1 else "Unsatisfied 😞"
                sentiment_color = "positive" if sentiment == 1 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Sentiment</h3>
                    <p class="{sentiment_color}">{sentiment_label}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Rating</h3>
                    <p class="neutral">{rating} {'⭐' * rating}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                confidence = "High" if abs(sentiment - 0.5) > 0.3 else "Medium"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Confidence</h3>
                    <p class="neutral">{confidence}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show cleaned text
            with st.expander("View Processed Text"):
                st.write("**Original Text:**")
                st.write(review_text)
                st.write("**Cleaned Text:**")
                st.write(cleaned_text)
        else:
            st.error("Please enter a review to analyze.")

def show_batch_analysis_page(tfidf, sentiment_model, rating_model):
    """Display batch analysis page"""
    st.markdown("## 📁 Batch Review Analysis")
    
    st.markdown("Upload a CSV file with hotel reviews or enter multiple reviews:")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload CSV file (should have a 'Review' column)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if 'Review' in batch_df.columns:
                st.success(f"Successfully loaded {len(batch_df)} reviews!")
                
                # Analyze all reviews
                if st.button("Analyze All Reviews", type="primary"):
                    with st.spinner("Analyzing reviews..."):
                        results = []
                        for idx, row in batch_df.iterrows():
                            review = row['Review']
                            sentiment, rating, _ = predict_sentiment(
                                review, tfidf, sentiment_model, rating_model
                            )
                            results.append({
                                'Review': review,
                                'Sentiment': 'Satisfied' if sentiment == 1 else 'Unsatisfied',
                                'Predicted_Rating': rating
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("CSV file must contain a 'Review' column.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Manual batch input
    st.markdown("### Or enter multiple reviews manually:")
    
    reviews_input = st.text_area(
        "Enter reviews (one per line):",
        placeholder="Review 1\nReview 2\nReview 3\n...",
        height=200
    )
    
    if st.button("Analyze Manual Reviews", type="primary"):
        if reviews_input.strip():
            reviews = [review.strip() for review in reviews_input.split('\n') if review.strip()]
            
            with st.spinner("Analyzing reviews..."):
                results = []
                for review in reviews:
                    sentiment, rating, _ = predict_sentiment(
                        review, tfidf, sentiment_model, rating_model
                    )
                    results.append({
                        'Review': review,
                        'Sentiment': 'Satisfied' if sentiment == 1 else 'Unsatisfied',
                        'Predicted_Rating': rating
                    })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.markdown("### 📊 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    satisfied_count = len(results_df[results_df['Sentiment'] == 'Satisfied'])
                    st.metric("Satisfied Reviews", satisfied_count)
                
                with col2:
                    unsatisfied_count = len(results_df[results_df['Sentiment'] == 'Unsatisfied'])
                    st.metric("Unsatisfied Reviews", unsatisfied_count)
                
                with col3:
                    avg_rating = results_df['Predicted_Rating'].mean()
                    st.metric("Average Rating", f"{avg_rating:.2f}")
        else:
            st.error("Please enter reviews to analyze.")

def show_model_performance_page(sentiment_model, rating_model, X_test_s, y_test_s, X_test_r, y_test_r):
    """Display model performance metrics"""
    st.markdown("## 📈 Model Performance")
    
    # Sentiment model performance
    st.markdown("### Sentiment Classification Model")
    
    y_pred_sentiment = sentiment_model.predict(X_test_s)
    sentiment_accuracy = accuracy_score(y_test_s, y_pred_sentiment)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{sentiment_accuracy:.3f}")
    
    with col2:
        sentiment_report = classification_report(y_test_s, y_pred_sentiment, output_dict=True)
        precision = sentiment_report['weighted avg']['precision']
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = sentiment_report['weighted avg']['recall']
        st.metric("Recall", f"{recall:.3f}")
    
    # Sentiment confusion matrix
    cm_sentiment = confusion_matrix(y_test_s, y_pred_sentiment)
    fig = px.imshow(cm_sentiment, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Unsatisfied', 'Satisfied'],
                    y=['Unsatisfied', 'Satisfied'],
                    title="Sentiment Classification Confusion Matrix",
                    color_continuous_scale='Blues',
                    text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Rating model performance
    st.markdown("### Rating Prediction Model")
    
    y_pred_rating = rating_model.predict(X_test_r)
    rating_accuracy = accuracy_score(y_test_r, y_pred_rating)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{rating_accuracy:.3f}")
    
    with col2:
        rating_report = classification_report(y_test_r, y_pred_rating, output_dict=True)
        precision = rating_report['weighted avg']['precision']
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = rating_report['weighted avg']['recall']
        st.metric("Recall", f"{recall:.3f}")
    
    # Rating confusion matrix
    cm_rating = confusion_matrix(y_test_r, y_pred_rating)
    fig = px.imshow(cm_rating,
                    labels=dict(x="Predicted Rating", y="Actual Rating", color="Count"),
                    x=[1, 2, 3, 4, 5],
                    y=[1, 2, 3, 4, 5],
                    title="Rating Prediction Confusion Matrix",
                    color_continuous_scale='Greens',
                    text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 