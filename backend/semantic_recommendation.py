import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration
CSV_PATH = "data/amazon_products.csv"
PERSIST_DIR = "db/"
COLLECTION_NAME = "amazon_products"

# Load embedding model
print("Loading SentenceTransformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up ChromaDB client
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

# Create or retrieve collection
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Load and process CSV
print(f"Loading CSV...")
df = pd.read_csv(CSV_PATH)

# Basic cleaning
print("Cleaning data...")
df = df.dropna(subset=['product_name', 'about_product'])
df = df.drop_duplicates(subset=['about_product'])

# Optional: enrich the text
print("Enriching product descriptions...")
df["text"] = df["product_name"] + " - " + df["about_product"]

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# Prepare metadata
ids = [str(i) for i in df.index]
metadatas = df[["product_name", "category"]].to_dict(orient='records')

# Load into vector DB if not already present
if collection.count() == 0:
    print("Saving vectors to ChromaDB...")
    collection.add(
        documents=df["text"].tolist(),
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"{len(ids)} products successfully stored in the vector database.")
else:
    print(f"Collection already contains {collection.count()} items. No new data was added.")

print("Semantic recommendation setup complete.")
