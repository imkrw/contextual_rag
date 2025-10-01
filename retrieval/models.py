from __future__ import annotations
from typing import Protocol, Sequence, TypedDict


class PageDocument(TypedDict, total=False):
    pageContent: str


class Metadata(TypedDict, total=False):
    document: PageDocument
    pageContent: str
    text: str
    content: str


class Match(TypedDict, total=False):
    id: str
    score: float
    metadata: Metadata


class PineconeIndex(Protocol):
    def query(
        self,
        *,
        namespace: str,
        vector: Sequence[float],
        top_k: int,
        include_values: bool,
        include_metadata: bool,
    ) -> object: ...
