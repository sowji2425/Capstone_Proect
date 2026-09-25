import os
import glob
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = os.getenv("CHROMA_DATA_PATH", "./chroma_db")
COLLECTION_NAME = "zepto_policies"

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def ingest_corpus(docs_dir: str = "./docs"):
    collection = get_chroma_collection()
    file_paths = sorted(glob.glob(os.path.join(docs_dir, "doc_*.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No document files found in {docs_dir}")

    documents = []
    ids = []
    metadatas = []

    for path in file_paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        documents.append(content)
        ids.append(doc_id)
        metadatas.append({"source": doc_id, "file_path": path})

    # Upsert to prevent duplicate entries on rerun
    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Successfully ingested {len(ids)} documents into ChromaDB collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    ingest_corpus()
