import chromadb
from sentence_transformers import SentenceTransformer

PERSIST_DIR = "db/"
COLLECTION_NAME = "amazon_products"

# Load model and persistent ChromaDB
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def search_similar_products(query: str, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    suggestions = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        suggestions.append({
            "product_name": meta.get("product_name", "N/A"),
            "category": meta.get("category", "N/A"),
            "actual_price": meta.get("actual_price", "N/A"),
            "img_link": meta.get("img_link", ""),
            "chunk_index": meta.get("chunk_index", "N/A"),
            "description_chunk": doc
        })

    return suggestions
