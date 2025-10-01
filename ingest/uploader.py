import uuid
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from openai import OpenAI
from core.embeddings import get_embedding
from core.presets import (
    ALLOWED_NAMESPACES,
    SITUATE_SYSTEM_PROMPT,
    build_situated_chunk_instructions,
)
from retrieval.index import get_pinecone_index


def _extract_upload_entry(item: object) -> Optional[Tuple[Path, str]]:
    if item is None:
        return None

    if hasattr(item, "name"):
        temp_path = Path(getattr(item, "name"))
        original_filename = getattr(item, "orig_name", temp_path.name)
        return temp_path, original_filename

    if isinstance(item, dict) and "name" in item:
        temp_path = Path(item["name"])
        original_filename = item.get("orig_name", temp_path.name)
        return temp_path, original_filename

    temp_path = Path(str(item))
    return temp_path, temp_path.name


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{path}: file must be UTF-8 encoded") from exc


def _find_split_point(s: str, preferred_at: int) -> Optional[int]:
    split_point = s.rfind("\n\n", 0, preferred_at)
    if split_point != -1:
        return split_point
    split_point = s.rfind(". ", 0, preferred_at)
    if split_point != -1:
        return split_point + 1
    split_point = s.rfind(" ", 0, preferred_at)
    if split_point != -1:
        return split_point
    return None


def _chunk_text(
    text: str, chunk_size: int = 2000, overlap: int = 0
) -> List[Tuple[str, int, int]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    chunks: List[Tuple[str, int, int]] = []
    cursor = 0
    text_length = len(cleaned)
    min_chunk_length = int(chunk_size * 0.5)
    tail_threshold = int(chunk_size * 0.2)

    while cursor < text_length:
        preferred_end = min(cursor + chunk_size, text_length)
        window = cleaned[cursor:preferred_end]
        relative_split = _find_split_point(window, len(window))
        if relative_split is None or relative_split < min_chunk_length:
            boundary = preferred_end
        else:
            boundary = cursor + relative_split

        boundary = min(boundary, text_length)
        chunk_text = cleaned[cursor:boundary].strip()
        if chunk_text:
            chunks.append((chunk_text, cursor, boundary))

        next_cursor = max(0, boundary - overlap)
        if 0 < (text_length - next_cursor) < tail_threshold:
            tail_text = cleaned[next_cursor:text_length].strip()
            if tail_text:
                chunks.append((tail_text, next_cursor, text_length))
            break

        cursor = next_cursor

    return chunks


def _char_index_to_line(index: int, n_positions: List[int]) -> int:
    left, right = 0, len(n_positions) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if n_positions[mid] <= index:
            left = mid
        else:
            right = mid - 1
    return left + 1


def _generate_situated_chunk(client: OpenAI, document_text: str, chunk: str) -> str:
    user_prompt = build_situated_chunk_instructions(document_text, chunk)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SITUATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return " ".join(text.splitlines()).strip()
    except Exception:
        return f"Document excerpt : {chunk.strip()}"


def _upsert_chunks(
    namespace: str,
    file_path: Path,
    client: OpenAI,
) -> int:
    index = get_pinecone_index()

    full_text = _read_text(file_path)
    spans = _chunk_text(full_text)
    if not spans:
        return 0

    total = 0
    vectors_batch = []
    n_positions: List[int] = [-1]
    for idx, ch in enumerate(full_text):
        if ch == "\n":
            n_positions.append(idx)
    n_positions.append(len(full_text))

    for piece, start_char, end_char in spans:
        material = _generate_situated_chunk(client, full_text, piece)
        embedding = get_embedding(client, material)
        try:
            vector_id = str(uuid.uuid4())
        except Exception:
            vector_id = f"{file_path.stem}-{start_char}-{end_char}"

        line_from = _char_index_to_line(start_char, n_positions)
        line_to = _char_index_to_line(max(end_char - 1, start_char), n_positions)

        vectors_batch.append(
            {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": material,
                    "blobType": "text/plain",
                    "source": "file",
                    "file": file_path.name,
                    "file_path": str(file_path),
                    "loc.lines.from": line_from,
                    "loc.lines.to": line_to,
                },
            }
        )
        if len(vectors_batch) >= 32:
            index.upsert(namespace=namespace, vectors=vectors_batch)
            total += len(vectors_batch)
            vectors_batch = []

    if vectors_batch:
        index.upsert(namespace=namespace, vectors=vectors_batch)
        total += len(vectors_batch)

    return total


def _process_files(
    namespace: str,
    file_paths: Sequence[Path],
    client: OpenAI,
) -> Tuple[int, List[str]]:
    total_vectors = 0
    log_messages: List[str] = []

    for file_path in file_paths:
        if not file_path.exists():
            log_messages.append(f"{file_path}: not found")
            continue

        if file_path.suffix.lower() != ".txt":
            log_messages.append(f"{file_path.name}: skipped (unsupported file type)")
            continue

        try:
            count = _upsert_chunks(namespace, file_path, client)
        except Exception as ex:
            log_messages.append(f"{file_path.name}: failed ({ex})")
            continue

        total_vectors += count
        log_messages.append(f"- {file_path.name}: {count} chunk(s) uploaded")

    return total_vectors, log_messages


def uploader(
    namespace: Optional[str],
    uploaded_files: Optional[List[object]],
    client: OpenAI,
) -> str:
    if namespace is None or namespace not in ALLOWED_NAMESPACES:
        return "Select a namespace before uploading."

    if not uploaded_files:
        return "No files received."

    original_name_by_temp: dict[str, str] = {}
    paths: List[Path] = []
    for item in uploaded_files:
        entry = _extract_upload_entry(item)
        if entry is None:
            continue

        path, display = entry
        paths.append(path)
        original_name_by_temp[path.name] = display

    if not paths:
        return "No valid files received."

    paths = list(dict.fromkeys(paths))
    total, logs = _process_files(namespace, paths, client)

    pretty_logs: List[str] = []
    for line in logs:
        pretty_line = line
        for temp_name, original_name in original_name_by_temp.items():
            pretty_line = pretty_line.replace(temp_name, original_name)
        pretty_logs.append(pretty_line)

    pretty_logs.append(f"Done. Total chunks uploaded: {total}")
    return "\n".join(pretty_logs)
