import faiss

# import numpy as np
import logging
import os

# import pickle
from sentence_transformers import SentenceTransformer
from typing import List  # , Tuple

from .config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH
from .models import Chunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = None
        self.embeddings = None  # Store embeddings alongside the index
        logger.info(
            f"Initialized SentenceTransformer model: {embedding_model_name}"
        )

    def build_index(self, chunks: List[Chunk]):
        """Builds a FAISS index from text chunks."""
        if not chunks:
            logger.warning("No chunks provided to build index.")
            self.index = None
            self.embeddings = None
            return

        texts = [chunk.text for chunk in chunks]
        logger.info(f"Encoding {len(texts)} chunks for vector index.")
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        self.embeddings = embeddings  # Store embeddings

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)  # Add vectors to the index
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve(
        self, query: str, chunks: List[Chunk], k: int = 3
    ) -> List[Chunk]:
        """Retrieves top K chunks based on query similarity."""
        if self.index is None or not chunks:
            logger.warning("Vector index not built or no chunks available.")
            return []

        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, k)

        # Ensure indices are within bounds
        valid_indices = [i for i in indices[0] if 0 <= i < len(chunks)]
        results = [chunks[i] for i in valid_indices]
        logger.info(
            f"Retrieved {len(results)} chunks for query: '{query[:50]}...'"
        )
        return results

    def save_index(self, index_path: str = FAISS_INDEX_PATH):
        """Saves the FAISS index and embeddings to disk."""
        if self.index is not None:
            try:
                faiss.write_index(self.index, index_path)
                # Saving embeddings is optional but can be useful
                # np.save(index_path.replace('.bin', '_embeddings.npy'), self.embeddings)
                logger.info(f"FAISS index saved to {index_path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {index_path}: {e}")
        else:
            logger.warning("No FAISS index to save.")

    def load_index(
        self, index_path: str = FAISS_INDEX_PATH, chunks: List[Chunk] = None
    ):
        """Loads the FAISS index from disk."""
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                # Loading embeddings if saved
                # embeddings_path = index_path.replace('.bin', '_embeddings.npy')
                # if os.path.exists(embeddings_path):
                #     self.embeddings = np.load(embeddings_path)
                logger.info(f"FAISS index loaded from {index_path}")
            except Exception as e:
                logger.error(
                    f"Error loading FAISS index from {index_path}: {e}"
                )
                self.index = None  # Ensure index is None if loading fails
                if chunks:
                    logger.info(
                        "Attempting to rebuild index as loading failed."
                    )
                    self.build_index(chunks)
        else:
            logger.warning(f"FAISS index file not found at {index_path}.")
            if chunks:
                logger.info("Building new index as file not found.")
                self.build_index(chunks)
            else:
                logger.warning("Cannot build index without chunks data.")


# Dependency to be used in FastAPI
def get_vector_store(chunks: List[Chunk]) -> VectorStore:
    """Provides a VectorStore instance, loading or building the index."""
    vector_store = VectorStore()
    # Attempt to load first, if fails or no file, build
    vector_store.load_index(chunks=chunks)
    # If loading failed and building also failed (e.g., no chunks), ensure index is None
    if vector_store.index is None and chunks:
        vector_store.build_index(chunks)
    return vector_store


if __name__ == '__main__':
    # Example usage for testing the vector_store module
    from ingestion import ingest_documents, load_chunks_data

    # Ensure documents are ingested and chunks data is available
    ingest_documents()
    all_chunks = load_chunks_data()

    if all_chunks:
        vector_store = get_vector_store(all_chunks)
        if vector_store.index:
            sample_query = "What are the product specifications?"
            top_chunks = vector_store.retrieve(sample_query, all_chunks)
            print("\nTop retrieved chunks:")
            for chunk in top_chunks:
                print(
                    f"- Doc: {chunk.doc}, Chunk ID: {chunk.chunk_id}, Text: {chunk.text[:100]}..."
                )

            # Example of saving and loading
            vector_store.save_index("test_faiss_index.bin")
            new_vector_store = VectorStore()
            new_vector_store.load_index("test_faiss_index.bin")
            if new_vector_store.index:
                print("\nSuccessfully loaded index.")
                loaded_top_chunks = new_vector_store.retrieve(
                    sample_query, all_chunks
                )
                print("Retrieved chunks after loading:")
                for chunk in loaded_top_chunks:
                    print(
                        f"- Doc: {chunk.doc}, Chunk ID: {chunk.chunk_id}, Text: {chunk.text[:100]}..."
                    )

        else:
            print("Vector index could not be built or loaded.")
    else:
        print("No chunks available to build vector index.")
