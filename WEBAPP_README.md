# 🏨 Hotel Sentiment Analyzer - Web Application

## Overview
This is an interactive web application built with Streamlit that provides sentiment analysis for hotel reviews. The app uses machine learning models to predict sentiment (satisfied/unsatisfied) and star ratings (1-5) from hotel review text.

## Features

### 🏠 Home Page
- Overview dashboard with key metrics
- Quick insights with interactive visualizations
- Feature highlights

### 📊 Data Analysis
- Interactive data visualizations using Plotly
- Rating distribution analysis
- Sentiment distribution charts
- Review length analysis
- Dataset statistics

### 🔍 Single Review Analysis
- Real-time sentiment analysis
- Star rating prediction
- Confidence scoring
- Text preprocessing visualization

### 📁 Batch Analysis
- CSV file upload support
- Manual batch review input
- Download results as CSV
- Summary statistics

### 📈 Model Performance
- Model accuracy metrics
- Confusion matrices
- Precision and recall scores
- Performance comparison

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "hotel sentiment"
   ```

2. **Run the deployment script**
   ```bash
   ./deploy.sh
   ```

3. **Start the application**
   ```bash
   source venv/bin/activate
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Manual Setup

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add web application"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Connect your repository
   - Deploy automatically

### Other Platforms

The app can also be deployed on:
- **Heroku**: Use the provided `requirements.txt`
- **Docker**: Create a Dockerfile (see below)
- **AWS/GCP**: Use container services

## File Structure

```
hotel sentiment/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── deploy.sh             # Deployment script
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── main.ipynb            # Original Jupyter notebook
├── tripadvisor_hotel_reviews.csv  # Dataset
├── README.md             # Original project README
└── WEBAPP_README.md      # This file
```

## Dependencies

- **Streamlit**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **NLTK**: Natural language processing
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations

## Usage Examples

### Single Review Analysis
1. Navigate to "🔍 Single Review" page
2. Enter a hotel review like: "The staff was friendly and the room was clean!"
3. Click "Analyze Review"
4. View sentiment and rating predictions

### Batch Analysis
1. Navigate to "📁 Batch Analysis" page
2. Upload a CSV file with a "Review" column
3. Or enter multiple reviews manually
4. Click "Analyze All Reviews"
5. Download results as CSV

### Data Insights
1. Navigate to "📊 Analysis" page
2. Explore interactive visualizations
3. View dataset statistics
4. Analyze review patterns

## Model Information

### Sentiment Classification
- **Model**: Logistic Regression
- **Features**: TF-IDF vectorization (3000 features)
- **Accuracy**: ~90%
- **Classes**: Satisfied (4-5 stars) vs Unsatisfied (1-3 stars)

### Rating Prediction
- **Model**: Logistic Regression
- **Features**: TF-IDF vectorization (3000 features)
- **Accuracy**: ~63%
- **Classes**: 1-5 star ratings

## Customization

### Adding New Models
1. Train your model in a separate script
2. Save it using `pickle` or `joblib`
3. Load it in `app.py`
4. Update the prediction functions

### Styling
- Modify the CSS in the `st.markdown` section
- Update colors and fonts
- Add custom components

### Data Sources
- Replace `tripadvisor_hotel_reviews.csv` with your dataset
- Update column names in the code
- Modify preprocessing as needed

## Troubleshooting

### Common Issues

1. **NLTK Data Not Found**
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

2. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. **Memory Issues**
   - Reduce `max_features` in TfidfVectorizer
   - Use smaller dataset for testing

4. **Deployment Issues**
   - Check all files are committed to Git
   - Verify `requirements.txt` is up to date
   - Ensure dataset is included in repository

### Performance Tips

- Use `@st.cache_data` for data loading
- Use `@st.cache_resource` for model loading
- Optimize text preprocessing
- Consider using lighter models for production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the original notebook (`main.ipynb`)
- Open an issue in the repository

---

**Happy Analyzing! 🏨✨** 