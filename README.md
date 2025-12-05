# GenAI-Powered Customer Review Insights & Sentiment Engine
AI-powered customer review insights and sentiment engine for Amazon product reviews using NLP, LLMs, and interactive analytics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app) <!-- Replace with your actual deployed URL if available -->

## Overview

This project is an end-to-end AI-driven analytics pipeline for extracting insights from customer reviews in the retail/e-commerce domain. It processes Amazon product reviews to analyze sentiment, identify common themes, detect pain points, and suggest product improvements. Built using free tools and resources, it demonstrates skills in Generative AI (GenAI), Natural Language Processing (NLP), data preprocessing, machine learning, and interactive visualization.

The pipeline is automated in Python and culminates in a user-friendly Streamlit dashboard for exploring insights. This is ideal for CX teams, product managers, or data analysts to understand voice-of-customer data at scale.

This repository serves as a portfolio piece showcasing integration of traditional NLP with modern LLMs (via HuggingFace) for business intelligence.

## Features

- **Data Ingestion**: Loads public Amazon review datasets (CSV format).
- **Text Preprocessing**: Cleans and prepares review text using NLTK and spaCy (lowercasing, tokenization, stopword removal, lemmatization).
- **Sentiment Analysis**: Classifies reviews as POSITIVE/NEGATIVE using a pre-trained DistilBERT model.
- **Theme Extraction**: Identifies key topics via Latent Dirichlet Allocation (LDA) for unsupervised clustering.
- **Pain Points & Suggestions**: Uses zero-shot classification with BART to detect complaints and improvement ideas.
- **Interactive Dashboard**: Streamlit app with filters, charts (sentiment distribution, themes, pain points, suggestions), and review browsing.
- **GPU Acceleration**: Optimized for Colab's T4 GPU for efficient model inference.
- **Free Resources Only**: No paid APIs or tools (e.g., HuggingFace free models, Kaggle datasets).

## Tech Stack

- **Programming**: Python 3.x
- **Libraries**:
  - Data Handling: pandas
  - NLP: NLTK, spaCy, scikit-learn (for LDA)
  - AI Models: HuggingFace Transformers (DistilBERT for sentiment, BART for zero-shot)
  - Visualization: Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Environment**: Google Colab (for development; supports free GPU)
- **Deployment**: Streamlit Community Cloud (free tier)

## Dataset

This project uses a public Amazon reviews dataset from Kaggle (e.g., "Amazon US Reviews" subset, ~4915 rows for efficiency). Key columns include:
- `reviewText`: The raw review content.
- `overall`: Star rating (1-5).
- `asin`: Product ID for filtering.

Download from [Kaggle Amazon Reviews](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) or similar. Place the CSV in your project folder as `amazon_review.csv`.

Processed output is saved as `processed_reviews.csv` with added columns: `cleaned_review`, `sentiment`, `theme`, `theme_label`, `pain_point`, `suggestion`.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/GenAi-customer-review-sentiment-engine.git
   cd GenAi-customer-review-sentiment-engine
   ```

2. Install dependencies (use a virtual environment recommended):
   ```
   pip install pandas nltk spacy scikit-learn transformers matplotlib seaborn streamlit tqdm
   python -m spacy download en_core_web_sm
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   ```

3. Download and place your dataset CSV in the root folder.

## Usage

### Running in Google Colab (Development Mode)
1. Open `notebook.ipynb` in Colab.
2. Follow the steps: Load data, preprocess, extract insights, save `processed_reviews.csv`.
3. Write and test `app.py` for the dashboard.

### Running Locally (Dashboard)
1. Ensure `processed_reviews.csv` is in the same folder as `app.py`.
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Open http://localhost:8501 in your browser.
4. Use sidebar filters (sentiment, pain points) and explore sections.

### Deployment
- Deploy to Streamlit Cloud: Connect your GitHub repo at [share.streamlit.io](https://share.streamlit.io), select `app.py`.
- Share the public URL on your portfolio/LinkedIn.

## Project Structure

```
├── notebook.ipynb          # Main Colab/Jupyter notebook with pipeline code
├── app.py                  # Streamlit dashboard script
├── processed_reviews.csv   # Sample processed output (generate via notebook)
├── amazon_review.csv       # Input dataset (not included; download from Kaggle)
├── README.md               # This file
└── requirements.txt        # Dependencies (generate with pip freeze > requirements.txt)
```

## Screenshots

### Dashboard Overview

<img width="1759" height="884" alt="image" src="https://github.com/user-attachments/assets/adf82dd8-21e3-477f-b3ad-6655dcea41b0" />

### Sentiment Distribution Chart and Top Themes by Review Count

<img width="1891" height="664" alt="image" src="https://github.com/user-attachments/assets/e00f2b40-8bdf-432a-93c2-ec038fc4bb90" />

### Sentiment Breakdown by Theme

<img width="1793" height="749" alt="image" src="https://github.com/user-attachments/assets/ae997a14-2f13-4b4c-abf8-26574e9c47d2" />

### Common complaints and Suggestions for improvement

<img width="1805" height="759" alt="image" src="https://github.com/user-attachments/assets/9bb87d83-cc86-4210-a186-a4f46ae06e46" />

### Engine's Sentiment Review details

<img width="1774" height="586" alt="image" src="https://github.com/user-attachments/assets/8fd03e4b-8c50-40c7-8430-12285a25f41f" />

## Limitations & Improvements
- Dataset size: Tested on ~5k reviews; scale up with more RAM/GPU.
- Model Customization: Pain/suggestion labels are predefined; fine-tune for domain-specific accuracy.
- Enhancements: Add word clouds, real-time data scraping (ethical/legal considerations apply), or integrate with APIs for live reviews.

## Contributing
Contributions welcome! Fork the repo, create a branch, and submit a pull request. Focus on bug fixes, feature additions, or documentation improvements.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Contact
- [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GurionRamapoguSajeevan)
- [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rs-gurion/)
- [![Email](https://img.shields.io/badge/email-%23D14836.svg?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gurion7007@gmail.com)

Built as a personal project to showcase AI and analytics skills. Any feedback will be appreciated!
