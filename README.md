# 🏨 Hotel Sentiment Analysis Project

## Overview
This project performs sentiment analysis on hotel reviews using natural language processing (NLP) techniques. The analysis helps understand customer satisfaction and identify key factors affecting hotel ratings. The project includes both a comprehensive Jupyter notebook analysis and a modern, interactive web application for real-time sentiment analysis.

## 🚀 Live Demo
- **Web Application**: Interactive Streamlit app with real-time analysis
- **Local Development**: Run locally with `streamlit run app.py`
- **Cloud Deployment**: Ready for deployment on Streamlit Cloud, Heroku, or Docker

## ⚡ Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd "hotel sentiment"

# Quick setup and run
chmod +x deploy.sh
./deploy.sh
source venv/bin/activate
streamlit run app.py
```

**Open your browser and go to `http://localhost:8501`** 🎉

## Project Structure
```
hotel sentiment/
├── README.md                 # Project documentation
├── main.ipynb               # Main analysis notebook
├── sample.ipynb             # Sample/example notebook
├── tripadvisor_hotel_reviews.csv  # Hotel review dataset
├── venv/                    # Virtual environment
├── app.py                   # 🆕 Streamlit web application
├── requirements.txt         # 🆕 Python dependencies
├── deploy.sh               # 🆕 Deployment script
├── Dockerfile              # 🆕 Docker configuration
├── docker-compose.yml      # 🆕 Docker Compose setup
├── .streamlit/             # 🆕 Streamlit configuration
│   └── config.toml
├── WEBAPP_README.md        # 🆕 Web app documentation
└── DEPLOYMENT_GUIDE.md     # 🆕 Deployment instructions
```

## ✨ Features

### 🔬 Core Analysis
- **Data Analysis**: Comprehensive analysis of hotel review data
- **Sentiment Analysis**: NLP-based sentiment classification (90% accuracy)
- **Text Preprocessing**: Advanced text cleaning and normalization
- **Visualization**: Interactive charts and graphs for insights
- **Machine Learning**: ML models for sentiment and rating prediction

### 🌐 Web Application
- **Interactive Dashboard**: Modern, responsive UI with 5 different pages
- **Real-time Analysis**: Instant sentiment and rating predictions
- **Batch Processing**: Upload CSV files or enter multiple reviews
- **Interactive Visualizations**: Plotly charts with hover effects
- **Model Performance**: Detailed accuracy metrics and confusion matrices
- **Mobile Responsive**: Works perfectly on desktop and mobile devices

### 📊 Key Capabilities
- **Single Review Analysis**: Analyze individual hotel reviews instantly
- **Batch Analysis**: Process multiple reviews with CSV upload
- **Data Insights**: Interactive visualizations and statistics
- **Export Results**: Download analysis results as CSV
- **Performance Metrics**: Model accuracy, precision, and recall scores

## Prerequisites
- Python 3.11+
- pip (Python package installer)
- Git (for version control)
- Web browser (for accessing the web application)

## Installation & Setup

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd "hotel sentiment"

# Or download and extract the project files
```

### 2. Quick Setup (Recommended)
```bash
# Make the deployment script executable
chmod +x deploy.sh

# Run the automated setup script
./deploy.sh
```

### 3. Manual Setup (Alternative)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 🚀 Usage

### 🌐 Web Application (Recommended)
1. **Quick Start**:
   ```bash
   ./deploy.sh
   source venv/bin/activate
   streamlit run app.py
   ```

2. **Open in Browser**: Navigate to `http://localhost:8501`

3. **Application Features**:
   - 🏠 **Home**: Overview dashboard with key metrics and insights
   - 📊 **Analysis**: Interactive data visualizations and statistics
   - 🔍 **Single Review**: Real-time sentiment and rating analysis
   - 📁 **Batch Analysis**: Upload CSV files or enter multiple reviews
   - 📈 **Model Performance**: Detailed accuracy metrics and confusion matrices

### 📓 Jupyter Notebook Analysis
1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Start Jupyter Notebook:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

3. Open the notebooks:
   - `main.ipynb` - Main analysis notebook with complete workflow
   - `sample.ipynb` - Sample/example notebook

### 🧪 Testing the Application
```bash
# Run the test script to verify models
python test_app.py
```

### 📁 Data Files
- `tripadvisor_hotel_reviews.csv` - Hotel review dataset (20,491 reviews)

## 📦 Dependencies

### Core Libraries
- **pandas** (2.3.1) - Data manipulation and analysis
- **numpy** (2.3.2) - Numerical computing
- **matplotlib** (3.10.3) - Plotting and visualization
- **seaborn** (0.13.2) - Statistical data visualization

### 🌐 Web Application
- **streamlit** (1.28.1) - Web framework for data apps
- **plotly** (5.17.0) - Interactive visualizations

### 🔤 NLP Libraries
- **nltk** (3.9.1) - Natural Language Toolkit
- **spacy** (3.8.7) - Advanced NLP library

### 🤖 Machine Learning
- **scikit-learn** (1.7.1) - Machine learning algorithms
- **scipy** (1.16.0) - Scientific computing

## 🔄 Project Workflow
1. **Data Loading**: Import and examine the hotel review dataset
2. **Data Preprocessing**: Clean and prepare text data
3. **Exploratory Data Analysis**: Understand data patterns and distributions
4. **Feature Engineering**: Extract relevant features from text
5. **Model Development**: Build and train sentiment analysis models
6. **Evaluation**: Assess model performance
7. **Visualization**: Create insightful visualizations
8. **Results Analysis**: Interpret findings and draw conclusions
9. **🌐 Web Application**: Deploy interactive web interface
10. **📊 User Interface**: Provide real-time analysis capabilities

## 📊 Model Performance
- **Sentiment Classification**: 90.0% accuracy
- **Rating Prediction**: 62.5% accuracy
- **Real-time Processing**: Instant predictions
- **Batch Processing**: Handle multiple reviews efficiently

## 🚀 Deployment

### Quick Deployment Options
- **Streamlit Cloud**: Free hosting with automatic deployment
- **Heroku**: Cloud platform deployment
- **Docker**: Containerized deployment
- **AWS/GCP**: Enterprise cloud deployment

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License
This project is for educational and research purposes.

## 📞 Contact
For questions or support, please open an issue in the repository.

## 📚 Documentation
- **`WEBAPP_README.md`**: Detailed web application documentation
- **`DEPLOYMENT_GUIDE.md`**: Comprehensive deployment instructions
- **`main.ipynb`**: Original analysis notebook
- **`test_app.py`**: Test script to verify model functionality

## 🙏 Acknowledgments
- TripAdvisor for providing the review dataset
- Open-source community for the excellent NLP and ML libraries
- Streamlit for the amazing web framework
- Plotly for interactive visualizations 