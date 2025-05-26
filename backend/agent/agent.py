import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

class RAGAgent:
    def __init__(self):
        self.gemini_llm = None
        self.hf_llm = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize LangChain models"""
        if GEMINI_TOKEN:
            try:
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=GEMINI_TOKEN,
                    temperature=0.4
                )
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")
        
        if HF_TOKEN:
            try:
                self.hf_llm = HuggingFaceEndpoint(
                    repo_id="HuggingFaceH4/zephyr-7b-beta",
                    huggingfacehub_api_token=HF_TOKEN,
                    temperature=0.4,
                    max_new_tokens=300
                )
            except Exception as e:
                print(f"Failed to initialize Hugging Face: {e}")
    
    def get_llm(self, model: str):
        """Get the appropriate LLM"""
        if model == "gemini" and self.gemini_llm:
            return self.gemini_llm
        elif model == "huggingface" and self.hf_llm:
            return self.hf_llm
        else:
            raise ValueError(f"Model {model} not available")
    
    def query_with_memory(
        self,
        user_input: str,
        product_info: str,
        web_results: str,
        memory: ConversationBufferWindowMemory,
        model: str = "gemini"
    ) -> str:
        """Query LLM with RAG context and conversation memory using LangChain"""
        try:
            llm = self.get_llm(model)
            
            # Create prompt template with memory placeholder
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert assistant with access to product database and web search results.

                Product Information:
                {product_info}

                Web Search Results:  
                {web_results}

                IMPORTANT INSTRUCTIONS:
                - Always review the conversation history before responding
                - When user asks follow-up questions like "which one" or "from those you mentioned", refer back to your previous responses
                - If you mentioned specific products/items earlier, reference them directly
                - Use information from product database, web search, AND previous conversation
                - Be consistent with what you said before"""),
                                
                # LangChain automatically injects conversation history here
                MessagesPlaceholder(variable_name="chat_history"),
                
                ("human", "{user_input}")
            ])
            
            # Create the chain
            chain = prompt | llm | StrOutputParser()
            
            # Get chat history from memory
            chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
            
            # Invoke chain with all context
            response = chain.invoke({
                "user_input": user_input,
                "product_info": product_info,
                "web_results": web_results,
                "chat_history": chat_history
            })
            
            return response
            
        except Exception as e:
            print(f"Error in query_with_memory: {e}")
            return f"Error processing request with {model} model"

# Global agent instance
agent = RAGAgent()

# Main function for RAG + Memory queries
def query_llm_with_context(
    user_input: str,
    product_info: str = "",
    web_results: str = "",
    memory: Optional[ConversationBufferWindowMemory] = None,
    model: str = "gemini"
) -> str:
    """
    Main function to query LLM with RAG context and memory using LangChain
    """
    if memory:
        return agent.query_with_memory(
            user_input=user_input,
            product_info=product_info,
            web_results=web_results,
            memory=memory,
            model=model
        )
    else:
        # Fallback without memory - create simple chain
        try:
            llm = agent.get_llm(model)
            simple_prompt = f"""Based on the following information, answer the question:

            Product Info: {product_info}
            Web Results: {web_results}
            Question: {user_input}

            Answer:"""
            
            response = llm.invoke(simple_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Error in fallback query: {e}")
            return "Error processing request"

# Legacy functions for backward compatibility
def query_llm(user_input: str, model: str = "huggingface") -> str:
    """Legacy function - kept for backward compatibility"""
    if model == "gemini":
        return query_gemini(user_input)
    return query_huggingface(user_input)

def query_gemini(user_input: str) -> str:
    """Legacy Gemini function"""
    if not GEMINI_TOKEN:
        return "Gemini API key is missing. Check your .env and variable name."

    import requests
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_TOKEN}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": user_input}]}]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("Gemini API error:", e)
        return "Error fetching response from Gemini"

def query_huggingface(user_input: str) -> str:
    """Legacy Hugging Face function"""
    import requests
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"User: {user_input}\nAssistant:",
        "parameters": {"max_new_tokens": 200, "temperature": 0.4}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        generated = response.json()
        if isinstance(generated, list):
            return generated[0]["generated_text"].split("Assistant:")[-1].strip()
        return "No response received from the model."
    except Exception as e:
        print("Hugging Face API error:", e)
        return "Error fetching response from Hugging Face"