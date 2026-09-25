import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from ingest import ingest_corpus
from graph import app_graph, QueryRequest, QueryResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure documents are ingested and vector collection is populated
    try:
        ingest_corpus()
    except Exception as e:
        print(f"Warning during ingestion startup: {e}")
    yield

app = FastAPI(
    title="Zepto Support Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": os.getenv("MOCK_LLM", "1")}

@app.post("/ask", response_model=QueryResponse)
def ask(payload: QueryRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    initial_state = {
        "query": payload.query,
        "intent": None,
        "retrieved_docs": None,
        "retrieved_ids": None,
        "final_response": None,
        "retry_count": 0
    }
    
    result = app_graph.invoke(initial_state)
    return result["final_response"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
