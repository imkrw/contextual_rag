from typing import List
from openai import OpenAI


def get_embedding(client: OpenAI, text: str) -> List[float]:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=1536,
    )
    return emb.data[0].embedding
