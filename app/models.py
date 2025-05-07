from pydantic import BaseModel
from typing import List, Dict, Any


class Chunk(BaseModel):
    doc: str
    chunk_id: int
    text: str


class RetrievalResult(BaseModel):
    branch: str
    result: str
    context: List[Chunk] = []
