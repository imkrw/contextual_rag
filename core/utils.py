import os
import retrieval.models as models
from typing import Optional, Sequence


def _extract_page_content(meta: models.Metadata | None) -> Optional[str]:
    if not meta or not isinstance(meta, dict):
        return None
    v = meta.get("text")
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def build_context(matches: Sequence[models.Match]) -> str:
    parts: list[str] = []
    for m in matches:
        meta = m.get("metadata") if isinstance(m, dict) else None
        text = _extract_page_content(meta)
        if text:
            parts.append(text)
    return "\n\n---\n\n".join(parts)


def ensure_environment_ready() -> None:
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    if not os.environ.get("PINECONE_INDEX"):
        missing.append("PINECONE_INDEX")

    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(f"Missing required environment variable(s): {joined}")
