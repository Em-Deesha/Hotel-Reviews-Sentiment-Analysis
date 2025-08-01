#!/bin/bash

echo "🏨 Hotel Sentiment Analyzer - Deployment Script"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

echo "✅ Setup complete!"
echo ""
echo "To run the application locally:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run Streamlit: streamlit run app.py"
echo ""
echo "To deploy to Streamlit Cloud:"
echo "1. Push your code to GitHub"
echo "2. Connect your repository to Streamlit Cloud"
echo "3. Deploy automatically!" 