from fastapi import FastAPI, Depends, HTTPException, status
from typing import List, Dict, Any
import logging

from app.config import FAISS_INDEX_PATH, CHUNKS_DATA_PATH
from app.ingestion import load_chunks_data, ingest_documents
from app.vector_store import VectorStore  # , get_vector_store
from app.agent import agent_route
from app.models import RetrievalResult, Chunk
import os
from contextlib import asynccontextmanager


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# @app.on_event("startup")
async def startup_event():
    """Load chunks and build/load vector index on application startup."""
    global all_chunks, vector_store_instance
    logger.info("startup_event: Starting up application...")

    # Ensure documents folder exists
    from app.config import DOCUMENTS_FOLDER

    if not os.path.exists(DOCUMENTS_FOLDER):
        logger.warning(
            f"Documents folder not found at {DOCUMENTS_FOLDER}. Creating it."
        )
        os.makedirs(DOCUMENTS_FOLDER)
        logger.warning(
            "Please add text documents to the 'documents/' folder and restart the application."
        )
        # We can't proceed without documents, but we won't crash.
        # The endpoints will return errors or empty results.
        return

    # Load chunks data
    all_chunks_obj = load_chunks_data()
    all_chunks = [
        chunk.model_dump() for chunk in all_chunks_obj
    ]  # Store as dicts for easier access

    # Initialize and load/build vector store
    vector_store_instance = VectorStore()
    vector_store_instance.load_index(
        chunks=all_chunks_obj
    )  # Pass Pydantic objects

    # If index still not built/loaded and we have chunks, build it
    if vector_store_instance.index is None and all_chunks_obj:
        logger.info("Index not loaded, building index from chunks.")
        vector_store_instance.build_index(all_chunks_obj)

    # Save the index after building/loading if it exists
    if vector_store_instance.index:
        vector_store_instance.save_index()
    else:
        logger.warning(
            "Vector index could not be initialized. RAG functionality will not work."
        )

    logger.info("Application startup complete.")


# @app.on_event("shutdown")
async def shutdown_event():
    """Save vector index on application shutdown."""
    global vector_store_instance
    logger.info("shutdown_event:Shutting down application...")
    if vector_store_instance and vector_store_instance.index:
        vector_store_instance.save_index()
    logger.info("Application shutdown complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML mo`del
    await startup_event()
    yield
    # Clean up the ML models and release the resources
    await shutdown_event()


app = FastAPI(
    title="RAG with Agentic Workflow API",
    description="API for a RAG system with agent-based routing.",
    lifespan=lifespan,
)

# Global variables to hold chunks data and vector store instance
# This assumes the data is loaded once when the app starts
all_chunks: List[Dict[str, Any]] = []
vector_store_instance: VectorStore = None


# Dependency to provide chunks data
def get_chunks_data() -> List[Dict[str, Any]]:
    """Provides the loaded chunks data."""
    if not all_chunks:
        logger.warning("Chunks data not loaded.")
        # Optionally, you could re-ingest here, but it's better handled in startup
        # ingest_documents()
        # global all_chunks
        # all_chunks = load_chunks_data()
    return all_chunks


# Dependency to provide vector store instance
def get_vector_store_dependency(
    chunks: List[Dict[str, Any]] = Depends(get_chunks_data),
) -> VectorStore:
    """Provides the initialized vector store instance."""
    global vector_store_instance
    if vector_store_instance is None or vector_store_instance.index is None:
        logger.error("Vector store not initialized or index not built.")
        # This case should ideally not happen if startup is successful
        # but provides a fallback/error state.
        # You could attempt to re-initialize here if needed, but it adds complexity.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not ready.",
        )

    # Ensure chunks data is available for retrieval if needed (though retrieve uses the instance's internal state)
    # Passing chunks here is mainly for the initial build/load in get_vector_store if used directly.
    # Since we manage the instance globally, we just return the initialized instance.
    return vector_store_instance


@app.get("/")
async def read_root():
    return {
        "message": "RAG with Agentic Workflow API is running. Go to /docs for API documentation."
    }


@app.post("/query", response_model=RetrievalResult)
async def process_query(
    query: str,
    vector_store: VectorStore = Depends(get_vector_store_dependency),
    chunks_data: List[Dict[str, Any]] = Depends(get_chunks_data),
):
    """
    Processes a user query using the RAG with Agentic Workflow.
    Routes the query to the appropriate tool or the RAG pipeline.
    """
    if not chunks_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document chunks not loaded. Please ensure documents are in the 'documents/' folder and restart the application.",
        )
    if vector_store.index is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector index not built or loaded. RAG functionality is unavailable.",
        )

    # Convert chunks_data back to Pydantic models for the agent_route function
    chunks_obj = [Chunk(**chunk_data) for chunk_data in chunks_data]

    try:
        result = agent_route(query, vector_store, chunks_obj)
        return result
    except Exception as e:
        logger.error(f"An error occurred while processing query '{query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )


@app.post("/ingest")
async def trigger_ingestion():
    """
    Triggers the document ingestion process.
    This will reload chunks and rebuild/reload the vector index.
    Use with caution in a production environment as it can be resource intensive.
    """
    global all_chunks, vector_store_instance
    logger.info("Triggering document ingestion via API.")
    try:
        # Re-ingest documents
        all_chunks_obj = ingest_documents()
        all_chunks = [chunk.model_dump() for chunk in all_chunks_obj]

        # Re-initialize and rebuild vector store
        vector_store_instance = VectorStore()
        vector_store_instance.build_index(all_chunks_obj)

        # Save the new index
        if vector_store_instance.index:
            vector_store_instance.save_index()
            message = f"Ingestion complete. {len(all_chunks)} chunks created and vector index rebuilt."
        else:
            message = f"Ingestion complete. {len(all_chunks)} chunks created, but vector index could not be built."
            logger.error("Vector index failed to build after ingestion.")

        return {"status": "success", "message": message}
    except Exception as e:
        logger.error(f"An error occurred during ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {e}",
        )


@app.get("/status")
async def get_status():
    """Provides the current status of the RAG system."""
    status_info = {
        "chunks_loaded": len(all_chunks) > 0,
        "num_chunks": len(all_chunks),
        "vector_index_ready": vector_store_instance is not None
        and vector_store_instance.index is not None,
        "vector_index_size": (
            vector_store_instance.index.ntotal
            if vector_store_instance and vector_store_instance.index
            else 0
        ),
        "llm_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "documents_folder": os.getenv("DOCUMENTS_FOLDER", "documents/"),
        "faiss_index_path": FAISS_INDEX_PATH,
        "chunks_data_path": CHUNKS_DATA_PATH,
    }
    return status_info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
