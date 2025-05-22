import os
import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def search_google_serpapi(query: str, num_results: int = 3) -> str:
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "num": num_results
        }

        res = requests.get("https://serpapi.com/search", params=params)
        res.raise_for_status()
        data = res.json()

        results = data.get("organic_results", [])[:num_results]
        return "\n".join(f"{r['title']}: {r['snippet']}" for r in results if 'title' in r and 'snippet' in r)

    except Exception as e:
        print("SerpAPI error:", e)
        return "⚠️ Error fetching Google search results"
