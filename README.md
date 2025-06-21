# 🎬 Movie Recommender System

This is a FastAPI-based web application that recommends movies similar to a user-given input using TF-IDF vectorization** and cosine similarity. The data is scraped from the IMDb Top 250 Movies list using Selenium, and recommendations can be filtered based on:
- 🎭 Genre
- 🎬 Director
- 🌟 Star
- 🧠 All combined

# 🛠️ Technologies Used

- Python 3
- FastAPI
- Selenium + WebDriver Manager (for IMDb scraping)
- pandas, scikit-learn (TF-IDF & cosine similarity)
- Tailwind CSS (frontend)
- Jinja2, HTML
- RapidFuzz (fuzzy matching)
- Uvicorn (ASGI server)

# 🚀 How It Works

1. IMDb Top 250 data is scraped using Selenium and saved as 'detail_movies.csv'.
2. Data is processed and cleaned using pandas.
3. TF-IDF vectorization is applied on selected features.
4. Cosine similarity is computed.
5. User inputs a movie and selects a feature filter.
6. The app returns movies with similarity > 30%.

# 💡 How to Run Locally

 ## 📦 Install Dependencies: pip install -r requirements.txt
 ## 🌐 Scrape IMDb data: python scrape.py
 ## 🚀 Start Fast-API app: uvicorn main:app --reload
 visit: http://127.0.0.1:8000
