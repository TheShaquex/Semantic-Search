import os
import shutil
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
CSV_PATH = "data/amazon_products.csv"
PERSIST_DIR = "db/"
COLLECTION_NAME = "amazon_products"

# --- Optional: Ask user to delete DB ---
delete_db = input("Do you wish to delete the previous ChromaDB database? Enter 1 for Yes, 0 for No: ")
if delete_db == "1" and os.path.exists(PERSIST_DIR):
    try:
        shutil.rmtree(PERSIST_DIR)
        print("Deleted ChromaDB directory.")
    except Exception as e:
        print(f"Error deleting DB directory: {e}")
        exit(1)

# --- Load model ---
print("Loading SentenceTransformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Set up ChromaDB client ---
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# --- Load and clean data ---
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["product_name", "about_product", "review_content", "actual_price"])
df = df.drop_duplicates(subset=["about_product"])

# --- Convert price to float from string ---
def clean_price(value):
    try:
        # Remove any non-digit or decimal characters (like ₹, $, commas)
        numeric_str = re.sub(r"[^\d.]", "", str(value))
        return float(numeric_str)
    except:
        return None

df["actual_price_clean"] = df["actual_price"].apply(clean_price)

# --- Enrich text for semantic embedding ---
print("Enriching product descriptions with reviews and price...")
df["text"] = (
    df["product_name"].fillna("") + " - " +
    df["about_product"].fillna("") + ". " +
    "Review Title: " + df["review_title"].fillna("") + ". " +
    "Review: " + df["review_content"].fillna("") + ". " +
    "Price: ₹" + df["actual_price_clean"].fillna("N/A").astype(str)
)

# --- Chunking with LangChain ---
print("Chunking product descriptions...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
chunked_texts = []
chunked_metadatas = []

for idx, row in df.iterrows():
    chunks = text_splitter.split_text(row["text"])
    for i, chunk in enumerate(chunks):
        chunked_texts.append(chunk)
        chunked_metadatas.append({
            "product_name": row["product_name"],
            "category": row["category"],
            "img_link": row.get("img_link", ""),
            "actual_price": row["actual_price_clean"] if pd.notnull(row["actual_price_clean"]) else 0.0,
            "chunk_index": i,
            "original_id": str(idx),
        })

# --- Generate embeddings ---
print("Generating embeddings...")
embeddings = model.encode(chunked_texts, show_progress_bar=True)

# --- Store in ChromaDB ---
if collection.count() == 0:
    print("Saving vectors to ChromaDB...")
    BATCH_SIZE = 5000
    for i in range(0, len(chunked_texts), BATCH_SIZE):
        collection.add(
            documents=chunked_texts[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE],
            metadatas=chunked_metadatas[i:i + BATCH_SIZE],
            ids=[f"{i + j}" for j in range(len(chunked_texts[i:i + BATCH_SIZE]))],
        )
    print(f"{len(chunked_texts)} chunks successfully stored in the vector database.")
else:
    print(f"Collection already contains {collection.count()} items. No new data was added.")

print("Semantic recommendation setup complete.")
