from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from recommend import load_data, get_recommendations
from models import MovieRequest
import pandas as pd
import re
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load movie titles
try:
    df = load_data()
    movie_titles = df["title"].dropna().str.title().unique().tolist()
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        logger.debug("Rendering index.html")
        return templates.TemplateResponse("index.html", {"request": request, "movie_titles": movie_titles})
    except Exception as e:
        logger.error(f"Error rendering index.html: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template error: {str(e)}")

@app.post("/recommend", response_class=HTMLResponse)
async def recommend_post(request: Request, title: str = Form(...), filter: str = Form("all")):
    # Validate input
    if not re.match(r'^[a-zA-Z0-9\s:.\'&]+$', title):
        logger.warning(f"Invalid title input: {title}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "movie_titles": movie_titles, "error": "Please enter a valid movie name (only letters, numbers, spaces, :, ., ', & allowed)."}
        )
    
    try:
        title_clean = title.strip().lower()
        if not title_clean:
            logger.warning("Empty title input")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "movie_titles": movie_titles, "error": "Please enter a movie title."}
            )
        title, recommendations = get_recommendations(title_clean, filter=filter)
        if not recommendations:
            logger.info(f"No recommendations for title: {title}")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "movie_titles": movie_titles, "error": f"No recommendations found for '{title.title()}'."}
            )
        logger.debug(f"Rendering results.html for title: {title}")
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "title": title.title(), "recommendations": recommendations}
        )
    except Exception as e:
        logger.error(f"Error in recommend_post: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "movie_titles": movie_titles, "error": f"Error: {str(e)}"}
        )

@app.post("/api/recommend")
async def recommend_api(title: str = Form(...), filter: str = Form("all")):
    """JSON endpoint for recommendations."""
    if not re.match(r'^[a-zA-Z0-9\s:.\'&]+$', title):
        logger.warning(f"Invalid title input: {title}")
        raise HTTPException(status_code=400, detail="Invalid movie name (only letters, numbers, spaces, :, ., ', & allowed).")
    
    try:
        title_clean = title.strip().lower()
        if not title_clean:
            logger.warning("Empty title input")
            raise HTTPException(status_code=400, detail="Movie title cannot be empty.")
        title, recommendations = get_recommendations(title_clean, filter=filter)
        if not recommendations:
            logger.info(f"No recommendations for title: {title}")
            return {"title": title.title(), "recommendations": []}
        return {"title": title.title(), "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error in recommend_api: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/movies")
async def get_movies(query: str = ""):
    """Return movie titles matching the query for autocomplete."""
    logger.debug(f"Fetching movies with query: {query}")
    try:
        if not query:
            return JSONResponse({"movie_titles": movie_titles})
        
        from rapidfuzz import fuzz
        matches = [
            {"title": title, "score": fuzz.partial_ratio(query.lower(), title.lower())}
            for title in movie_titles
        ]
        matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:10]
        return JSONResponse({"movie_titles": [m["title"] for m in matches]})
    except Exception as e:
        logger.error(f"Error in get_movies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}