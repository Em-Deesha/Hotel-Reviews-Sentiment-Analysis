# 🚀 Deployment Guide - Hotel Sentiment Analyzer

## Quick Start (Local Development)

### Option 1: Using the Deployment Script
```bash
# Make script executable (if not already)
chmod +x deploy.sh

# Run the deployment script
./deploy.sh

# Start the application
source venv/bin/activate
streamlit run app.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run the application
streamlit run app.py
```

## 🌐 Cloud Deployment Options

### 1. Streamlit Cloud (Recommended - Free)

**Step 1: Prepare Your Repository**
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit: Hotel Sentiment Analyzer"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/hotel-sentiment.git
git push -u origin main
```

**Step 2: Deploy to Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `yourusername/hotel-sentiment`
5. Set main file path: `app.py`
6. Click "Deploy"

**Step 3: Configure (Optional)**
- Add environment variables if needed
- Set up custom domain
- Configure app settings

### 2. Heroku Deployment

**Step 1: Create Heroku App**
```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login to Heroku
heroku login

# Create app
heroku create your-hotel-sentiment-app

# Add buildpack
heroku buildpacks:set heroku/python
```

**Step 2: Deploy**
```bash
# Push to Heroku
git push heroku main

# Open the app
heroku open
```

**Step 3: Scale (Optional)**
```bash
# Scale to 1 dyno
heroku ps:scale web=1
```

### 3. Docker Deployment

**Step 1: Build and Run with Docker**
```bash
# Build the image
docker build -t hotel-sentiment-app .

# Run the container
docker run -p 8501:8501 hotel-sentiment-app
```

**Step 2: Using Docker Compose**
```bash
# Start with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### 4. AWS/GCP Deployment

#### AWS Elastic Beanstalk
```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.11 hotel-sentiment

# Create environment
eb create hotel-sentiment-env

# Deploy
eb deploy
```

#### Google Cloud Run
```bash
# Install gcloud CLI
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/hotel-sentiment

# Deploy to Cloud Run
gcloud run deploy hotel-sentiment \
  --image gcr.io/PROJECT_ID/hotel-sentiment \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# App-specific variables
export MODEL_CACHE_ENABLED=true
export MAX_REVIEWS_PER_BATCH=1000
```

### Custom Domain Setup
1. **Streamlit Cloud**: Add custom domain in app settings
2. **Heroku**: `heroku domains:add yourdomain.com`
3. **AWS**: Configure Route 53 and load balancer
4. **GCP**: Set up Cloud Load Balancing

## 📊 Monitoring and Analytics

### Streamlit Cloud Analytics
- Built-in analytics dashboard
- User engagement metrics
- Performance monitoring

### Custom Monitoring
```python
# Add to app.py for custom metrics
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track usage
def log_prediction(review_length, prediction_time):
    logger.info(f"Prediction made: {review_length} chars, {prediction_time:.2f}s")
```

## 🔒 Security Considerations

### Environment Variables
```bash
# Never commit sensitive data
echo "*.env" >> .gitignore
echo "secrets/" >> .gitignore
```

### HTTPS Setup
- **Streamlit Cloud**: Automatic HTTPS
- **Heroku**: Automatic HTTPS
- **Custom**: Configure SSL certificates

### Rate Limiting
```python
# Add rate limiting to app.py
import streamlit as st
from datetime import datetime, timedelta

# Simple rate limiting
if 'last_request' not in st.session_state:
    st.session_state.last_request = datetime.now()

if datetime.now() - st.session_state.last_request < timedelta(seconds=1):
    st.error("Please wait before making another request.")
    st.stop()
```

## 🚨 Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find process using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8502
```

**2. Memory Issues**
```bash
# Increase memory limit
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Or optimize the app
# Reduce max_features in TfidfVectorizer
# Use smaller dataset for testing
```

**3. NLTK Data Not Found**
```bash
# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Or add to Dockerfile
RUN python -c "import nltk; nltk.download('stopwords')"
```

**4. Deployment Failures**
```bash
# Check logs
heroku logs --tail
docker-compose logs -f
streamlit logs

# Verify requirements
pip freeze > requirements.txt
```

### Performance Optimization

**1. Caching**
```python
@st.cache_data
def load_data():
    return pd.read_csv('tripadvisor_hotel_reviews.csv')

@st.cache_resource
def load_models():
    # Load and return models
    pass
```

**2. Lazy Loading**
```python
# Load models only when needed
if 'models_loaded' not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.models = load_models()
        st.session_state.models_loaded = True
```

**3. Batch Processing**
```python
# Process reviews in batches
def process_batch(reviews, batch_size=100):
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i+batch_size]
        # Process batch
        yield results
```

## 📈 Scaling Strategies

### Horizontal Scaling
- **Load Balancer**: Distribute traffic across multiple instances
- **Auto-scaling**: Automatically scale based on demand
- **CDN**: Cache static assets globally

### Vertical Scaling
- **Memory**: Increase RAM for larger datasets
- **CPU**: Use more powerful instances
- **Storage**: Add more disk space

### Database Scaling
- **Read Replicas**: Distribute read operations
- **Sharding**: Split data across multiple databases
- **Caching**: Use Redis for frequently accessed data

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit Cloud
      uses: streamlit/streamlit-deploy-action@v0.1.0
      with:
        streamlit_app_file: app.py
        streamlit_app_url: ${{ secrets.STREAMLIT_APP_URL }}
```

## 📞 Support

### Getting Help
1. **Documentation**: Check `WEBAPP_README.md`
2. **Issues**: Open GitHub issue
3. **Community**: Streamlit community forum
4. **Debugging**: Use `st.write()` for debugging

### Useful Commands
```bash
# Check app status
curl -I http://localhost:8501

# View logs
tail -f ~/.streamlit/logs/streamlit.log

# Profile performance
python -m cProfile -o profile.stats app.py

# Memory usage
ps aux | grep streamlit
```

---

**Happy Deploying! 🚀** 