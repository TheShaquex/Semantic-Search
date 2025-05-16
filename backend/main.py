from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.requests import QueryRequest
from agent.agent import query_llm
from search.web import search_google_serpapi
from agent.semantic_search import search_similar_products

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search(query: QueryRequest):
    user_question = query.user_input
    model = query.model

    # Semantic DB results
    semantic_results = search_similar_products(user_question)

    # Web search
    web_snippets = search_google_serpapi(user_question)

    # Combine for prompt
    product_info = "\n".join(
        f"{s['title']} ({s['category']}): {s['description']}" for s in semantic_results
    )
    full_prompt = f"""You are an expert assistant. Based on both product data and web results, answer the question:

Product Info:
{product_info}

Web Results:
{web_snippets}

User Question: {user_question}
Answer:"""

    response = query_llm(full_prompt, model=model)
    return {"result": response}
