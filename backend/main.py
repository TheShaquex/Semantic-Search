import uuid
from typing import Dict, Optional
from datetime import datetime, timedelta
from threading import Lock
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferWindowMemory

from models.requests import QueryRequest
from agent.agent import query_llm_with_context
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

# Session Management for LangChain Memory
class SessionManager:
    def __init__(self, cleanup_hours: int = 24):
        self.sessions: Dict[str, Dict] = {}
        self.cleanup_hours = cleanup_hours
        self.lock = Lock()
    
    def get_or_create_session(self, session_id: str = None, max_memory: int = 10) -> str:
        with self.lock:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    'memory': ConversationBufferWindowMemory(
                        k=max_memory,  # Keep last N exchanges
                        return_messages=True,
                        memory_key="chat_history"
                    ),
                    'created_at': datetime.now(),
                    'last_accessed': datetime.now()
                }
            else:
                # Update memory window size if changed
                self.sessions[session_id]['memory'].k = max_memory
                self.sessions[session_id]['last_accessed'] = datetime.now()
            
            return session_id
    
    def get_memory(self, session_id: str) -> Optional[ConversationBufferWindowMemory]:
        with self.lock:
            if session_id in self.sessions:
                return self.sessions[session_id]['memory']
            return None
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def cleanup_old_sessions(self):
        """Remove sessions older than cleanup_hours"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=self.cleanup_hours)
            to_remove = [
                sid for sid, data in self.sessions.items()
                if data['last_accessed'] < cutoff_time
            ]
            for sid in to_remove:
                del self.sessions[sid]
            print(f"Cleaned up {len(to_remove)} old sessions")

# Initialize session manager
session_manager = SessionManager()

@app.post("/search")
async def search(query: QueryRequest):
    user_question = query.user_input
    model = query.model
    
    # Get session info from request
    session_id = getattr(query, 'session_id', None)
    max_memory = getattr(query, 'max_memory', 10)
    
    # Get or create session
    session_id = session_manager.get_or_create_session(session_id, max_memory)
    
    # Get LangChain memory for this session
    memory = session_manager.get_memory(session_id)
    
    # Your existing search logic
    semantic_results = search_similar_products(user_question)
    web_snippets = search_google_serpapi(user_question)
    
    # Format results for LangChain
    product_info = "\n".join(
        f"{s['product_name']} ({s['category']}) - ${s['actual_price']}: {s['description_chunk']}\nImage: {s['img_link']}"
        for s in semantic_results
    )
    
    # Use LangChain agent with memory - this automatically handles conversation history
    response = query_llm_with_context(
        user_input=user_question,
        product_info=product_info,
        web_results=web_snippets,
        memory=memory,
        model=model
    )
    
    # Add this exchange to memory (LangChain way)
    if memory:
        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(response)
    
    # Return response with session info
    return {
        "result": response,
        "session_id": session_id,
        "semantic_results_count": len(semantic_results),
        "has_conversation_history": bool(memory and memory.chat_memory.messages)
    }

# Session management endpoints
@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    memory = session_manager.get_memory(session_id)
    if not memory:
        return {"error": "Session not found", "session_id": session_id}
    
    history = []
    messages = memory.chat_memory.messages
    
    for msg in messages:
        if hasattr(msg, 'type'):
            if msg.type == "human":
                history.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                history.append({"role": "assistant", "content": msg.content})
        else:
            # Fallback for different message types
            if "human" in str(type(msg)).lower():
                history.append({"role": "user", "content": msg.content})
            elif "ai" in str(type(msg)).lower():
                history.append({"role": "assistant", "content": msg.content})
    
    return {
        "session_id": session_id,
        "history": history,
        "message_count": len(messages)
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    success = session_manager.clear_session(session_id)
    return {
        "success": success, 
        "message": f"Session {session_id} cleared" if success else "Session not found"
    }

@app.post("/sessions/cleanup")
async def cleanup_sessions():
    """Manually trigger session cleanup"""
    session_manager.cleanup_old_sessions()
    return {"message": "Session cleanup completed"}

@app.get("/sessions/stats")
async def get_session_stats():
    """Get statistics about active sessions"""
    with session_manager.lock:
        total_sessions = len(session_manager.sessions)
        session_info = []
        
        for sid, data in session_manager.sessions.items():
            message_count = len(data['memory'].chat_memory.messages)
            session_info.append({
                "session_id": sid,
                "message_count": message_count,
                "created_at": data['created_at'].isoformat(),
                "last_accessed": data['last_accessed'].isoformat()
            })
    
    return {
        "total_sessions": total_sessions,
        "sessions": session_info
    }

# Background cleanup (runs every hour)
import threading
import time

def periodic_cleanup():
    while True:
        time.sleep(3600)  # Run every hour
        session_manager.cleanup_old_sessions()

cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)