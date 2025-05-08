import os
import nltk
import json
import logging
from typing import List, Dict  # , Any

from .config import DOCUMENTS_FOLDER, CHUNKS_DATA_PATH
from .models import Chunk
from nltk.tokenize import sent_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def load_documents(doc_folder: str) -> Dict[str, str]:
    """Loads text documents from a folder."""
    docs = {}
    if not os.path.exists(doc_folder):
        logger.warning(f"Documents folder not found: {doc_folder}")
        return docs
    for filename in os.listdir(doc_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(doc_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    docs[filename] = f.read()
                logger.info(f"Loaded document: {filename}")
            except Exception as e:
                logger.error(f"Error loading document {filename}: {e}")
    return docs


def chunk_text(text: str, max_sentences: int = 5) -> List[str]:
    """Chunks text into smaller pieces based on sentence count."""
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i : i + max_sentences])
        chunks.append(chunk)
    return chunks


def ingest_documents(
    doc_folder: str = DOCUMENTS_FOLDER,
    chunks_output_path: str = CHUNKS_DATA_PATH,
) -> List[Chunk]:
    """Ingests documents, chunks them, and saves the chunks."""
    docs = load_documents(doc_folder)
    all_chunks: List[Chunk] = []
    for doc_name, text in docs.items():
        chunks = chunk_text(text)
        for idx, chunk_text_content in enumerate(chunks):
            all_chunks.append(
                Chunk(doc=doc_name, chunk_id=idx, text=chunk_text_content)
            )
    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Save chunks data
    try:
        with open(chunks_output_path, 'w', encoding='utf-8') as f:
            json.dump([chunk.model_dump() for chunk in all_chunks], f, indent=4)
        logger.info(f"Chunks data saved to {chunks_output_path}")
    except Exception as e:
        logger.error(f"Error saving chunks data to {chunks_output_path}: {e}")

    return all_chunks


def load_chunks_data(chunks_data_path: str = CHUNKS_DATA_PATH) -> List[Chunk]:
    """Loads chunks data from a JSON file."""
    if not os.path.exists(chunks_data_path):
        logger.warning(
            f"Chunks data file not found: {chunks_data_path}. Running ingestion."
        )
        return ingest_documents()

    try:
        with open(chunks_data_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            all_chunks = [Chunk(**chunk_data) for chunk_data in chunks_data]
        logger.info(f"Loaded {len(all_chunks)} chunks from {chunks_data_path}")
        return all_chunks
    except Exception as e:
        logger.error(f"Error loading chunks data from {chunks_data_path}: {e}")
        logger.info("Attempting to re-ingest documents.")
        return ingest_documents()


if __name__ == '__main__':
    # Example usage for testing the ingestion module
    ingested_chunks = ingest_documents()
    print(f"Ingested {len(ingested_chunks)} chunks.")
    if ingested_chunks:
        print("First chunk:", ingested_chunks[0])

    loaded_chunks = load_chunks_data()
    print(f"Loaded {len(loaded_chunks)} chunks.")
