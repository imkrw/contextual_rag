import os
from pinecone import Pinecone


def get_pinecone_index():
    index_name = os.environ.get("PINECONE_INDEX")
    api_key = os.environ.get("PINECONE_API_KEY")
    if not index_name or not api_key:
        return None
    try:
        pc = Pinecone(api_key=api_key)
        return pc.Index(index_name)
    except Exception:
        return None
