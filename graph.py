import os
import json
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from ingest import get_chroma_collection
from prompts import SYSTEM_RAG_PROMPT_TEMPLATE, CLASSIFIER_PROMPT

# Pydantic Schema
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

# TypedDict State for LangGraph
class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_docs: Optional[List[str]]
    retrieved_ids: Optional[List[str]]
    final_response: Optional[QueryResponse]
    retry_count: int

def is_mock_mode() -> bool:
    val = os.getenv("MOCK_LLM", "1").strip().lower()
    return val in ("1", "true", "yes", "")

# ----------------- Helper for Real LLM Calls (Optional Extension) -----------------
def call_real_llm(prompt: str) -> str:
    """Calls Groq or compatible API if MOCK_LLM=0 is set."""
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing for MOCK_LLM=0")
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        temperature=0.0
    )
    return chat_completion.choices[0].message.content

# ----------------- Nodes -----------------
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]

def classify_intent(state: AgentState) -> AgentState:
    query = state["query"]
    
    if is_mock_mode():
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        try:
            resp = call_real_llm(CLASSIFIER_PROMPT.format(query=query)).strip().lower()
            intent = "policy_question" if "policy_question" in resp else "general_question"
        except Exception:
            # Fallback to keyword heuristic on connection failure
            query_lower = query.lower()
            intent = "policy_question" if any(k in query_lower for k in POLICY_KEYWORDS) else "general_question"

    return {**state, "intent": intent}

def retrieve_and_answer(state: AgentState) -> AgentState:
    query = state["query"]
    collection = get_chroma_collection()
    
    # Real retrieval runs in both mock and real mode
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_ids = results["ids"][0] if results["ids"] else []
    
    if is_mock_mode():
        top_snippet = retrieved_docs[0][:200] if retrieved_docs else "No content available."
        response = QueryResponse(
            answer=f"Based on the retrieved context: {top_snippet}",
            sources=retrieved_ids,
            confidence=1.0
        )
        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "retrieved_ids": retrieved_ids,
            "final_response": response
        }
    
    # Optional Real LLM Path with validation and up to 2 retries
    context_str = "\n\n".join([f"[{d_id}] {doc}" for d_id, doc in zip(retrieved_ids, retrieved_docs)])
    prompt = SYSTEM_RAG_PROMPT_TEMPLATE.format(context=context_str) + f"\nUser Question: {query}\nResponse:"
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            raw_output = call_real_llm(prompt)
            # Find and parse JSON block
            json_start = raw_output.find("{")
            json_end = raw_output.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                parsed = json.loads(raw_output[json_start:json_end])
            else:
                parsed = json.loads(raw_output)
            
            validated = QueryResponse(**parsed)
            return {
                **state,
                "retrieved_docs": retrieved_docs,
                "retrieved_ids": retrieved_ids,
                "final_response": validated
            }
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                prompt += f"\n\nCorrection instruction: Your previous output failed JSON/Pydantic validation with error: {str(e)}. Return ONLY a valid JSON object matching the exact schema."
            else:
                # Fallback response after retries exhausted
                fallback = QueryResponse(
                    answer="Error: Unable to generate valid structured response from model.",
                    sources=retrieved_ids,
                    confidence=0.0
                )
                return {
                    **state,
                    "retrieved_docs": retrieved_docs,
                    "retrieved_ids": retrieved_ids,
                    "final_response": fallback,
                    "retry_count": attempt
                }

def direct_answer(state: AgentState) -> AgentState:
    if is_mock_mode():
        response = QueryResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
        return {**state, "final_response": response}
    
    # Optional Real LLM Path for general queries
    try:
        raw_output = call_real_llm(f"You are a Zepto customer support bot. Answer politely: {state['query']}")
        response = QueryResponse(
            answer=raw_output.strip(),
            sources=[],
            confidence=0.9
        )
    except Exception:
        response = QueryResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    return {**state, "final_response": response}

# ----------------- Conditional Router -----------------
def route_intent(state: AgentState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

# ----------------- StateGraph Assembly -----------------
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)
    
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    
    return workflow.compile()

# Global compiled instance
app_graph = build_graph()
