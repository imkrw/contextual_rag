import retrieval.models as models
from typing import Sequence, List


def query_pinecone(
    index: models.PineconeIndex,
    embedding: Sequence[float],
    namespace: str,
    top_k: int = 5,
) -> List[models.Match]:

    try:
        res = index.query(
            namespace=namespace,
            vector=embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
        )
    except Exception:
        return []

    response_matches = getattr(res, "matches", None)
    if response_matches is None and isinstance(res, dict):
        response_matches = res.get("matches")

    normalized_matches: List[models.Match] = []
    for m in response_matches or []:
        if isinstance(m, dict):
            normalized_matches.append(
                {
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "metadata": m.get("metadata"),
                }
            )
        else:
            normalized_matches.append(
                {
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", None),
                }
            )
    return normalized_matches
