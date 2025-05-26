# In models/requests.py
from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    user_input: str
    model: str = "gemini"
    session_id: Optional[str] = None  # Add this
    max_memory: Optional[int] = 10    # Add this