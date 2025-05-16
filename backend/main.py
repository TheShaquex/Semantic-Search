from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.requests import QueryRequest
from agent.agent import query_llm
from search.web import search_google_serpapi


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

    web_snippets = search_google_serpapi(user_question)

    full_prompt = f"""Based on the following Google search results, answer the question:

{web_snippets}

User question: {user_question}
Answer:"""

    response = query_llm(full_prompt, model=model)
    return {"result": response}