import logging
from typing import List, Tuple

from .vector_store import VectorStore
from .llm import get_llm_answer
from .models import Chunk, RetrievalResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculator_tool(query: str) -> str:
    """Naive calculator tool: extracts simple arithmetic expressions."""
    logger.info(f"Using calculator tool for query: {query}")
    try:
        # A more robust approach would use a proper expression parser
        # This is very basic and susceptible to injection if not careful
        if "calculate" in query.lower():
            expression = query.lower().split("calculate", 1)[1].strip()
            # Basic validation to prevent arbitrary code execution
            if all(c in '0123456789.+-*/() ' for c in expression):
                result = eval(expression)
                return f"Calculator result: {result}"
            else:
                return "Invalid expression for calculator."
        else:
            return "Query does not contain 'calculate'."
    except Exception as e:
        logger.error(f"Calculator tool error for query '{query}': {e}")
        return f"Calculator error: Could not evaluate expression.: {e}"


def dictionary_tool(query: str) -> str:
    """Fake dictionary tool: provides dummy definitions."""
    logger.info(f"Using dictionary tool for query: {query}")
    try:
        if "define" in query.lower():
            # Extract the word to define (simple approach)
            parts = query.lower().split("define", 1)[1].strip().split()
            if not parts:
                return "Please specify a word to define."
            word = parts[0]

            # Dummy definition lookup
            definitions = {
                "example": "a representative form or pattern",
                "sample": "a small part or quantity intended to show what the whole is like",
                "fastapi": "a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
                "rag": "Retrieval-Augmented Generation: a technique that enhances language models by retrieving relevant information from an external knowledge base.",
                "agent": "In AI, an agent is a system that perceives its environment and takes actions to achieve its goals.",
            }
            definition = definitions.get(word.lower(), "Definition not found.")
            return f"Definition of {word}: {definition}"
        else:
            return "Query does not contain 'define'."
    except Exception as e:
        logger.error(f"Dictionary tool error for query '{query}': {e}")
        return f"Dictionary error: Could not process definition request.: {e}"


def agent_route(
    query: str, vector_store: VectorStore, chunks: List[Chunk]
) -> RetrievalResult:
    """Routes the query based on simple keyword matching."""
    query_lower = query.lower()

    if "calculate" in query_lower:
        logger.info("Agent routing to calculator tool.")
        result = calculator_tool(query)
        return RetrievalResult(branch="calculator", result=result)
    elif "define" in query_lower:
        logger.info("Agent routing to dictionary tool.")
        result = dictionary_tool(query)
        return RetrievalResult(branch="dictionary", result=result)
    else:
        logger.info("Agent routing to RAG pipeline.")
        # RAG Pipeline: retrieve context and call LLM
        context_chunks = vector_store.retrieve(query, chunks, k=3)
        answer = get_llm_answer(query, context_chunks)
        return RetrievalResult(
            branch="RAG", result=answer, context=context_chunks
        )


if __name__ == '__main__':
    # Example usage for testing the agent module
    from ingestion import ingest_documents, load_chunks_data
    from vector_store import VectorStore  # , get_vector_store

    # import os

    # Ensure documents are ingested and chunks data is available
    ingest_documents()
    all_chunks = load_chunks_data()

    # Initialize and load/build vector store
    vector_store = VectorStore()
    vector_store.load_index(chunks=all_chunks)
    if vector_store.index is None:
        print(
            "Warning: Vector index could not be built or loaded. RAG functionality may be limited."
        )

    print("\nTesting Agent with different queries:")

    queries_to_test = [
        "What is the warranty period?",
        "calculate 10 + 5 * (3 - 1)",
        "define FastAPI",
        "Tell me about the product features.",
        "define sample",
        "calculate 100 / 4",
    ]

    for test_query in queries_to_test:
        print(f"\n--- Query: {test_query} ---")
        result = agent_route(test_query, vector_store, all_chunks)
        print(f"Agent Branch: {result.branch}")
        print(f"Result: {result.result}")
        if result.branch == "RAG" and result.context:
            print("Retrieved Context Snippets:")
            for idx, chunk in enumerate(result.context, 1):
                print(f"  {idx}: {chunk.text[:80]}...")
