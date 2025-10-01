from typing import Dict
from agents import RunConfig


ENHANCER_PROMPTS: Dict[str, str] = {
    "general_hotel_information": (
        "You are an expert at enhancing general hotel information search queries.\n"
        "Your task is to reformulate the given general-hotel-related query to make it more precise and specific for general hotel information retrieval.\n"
        "Provide ONLY the enhanced query without any explanation."
    ),
    "room_services": (
        "You are an expert at enhancing room service search queries.\n"
        "Your task is to reformulate the given room service query to make it more precise and specific for room service information retrieval.\n"
        "Provide ONLY the enhanced query without any explanation."
    ),
    "hotel_policies": (
        "You are an expert at enhancing hotel policy search queries.\n"
        "Your task is to reformulate the given hotel policy query to make it more precise and specific for hotel policy information retrieval.\n"
        "Provide ONLY the enhanced query without any explanation."
    ),
    "local_hotel_information": (
        "You are an expert at enhancing local hotel information search queries.\n"
        "Your task is to reformulate the given local hotel information query to make it more precise and specific for local hotel information retrieval.\n"
        "Provide ONLY the enhanced query without any explanation."
    ),
    "hotel_facilities": (
        "You are an expert at enhancing hotel facilities information search queries.\n"
        "Your task is to reformulate the given hotel facilities information query to make it more precise and specific for hotel facilities information retrieval.\n"
        "Provide ONLY the enhanced query without any explanation."
    ),
}

INSTRUCTION_CLASSIFIER = (
    "You are an expert at classifying hotel-related queries.\n"
    "Classify the given query into exactly one of the following categories: "
    "general_hotel_information, room_services, hotel_policies, local_hotel_information, hotel_facilities, other_unrelated.\n"
    "Return ONLY the category name, without any explanation or additional text."
)

SITUATE_SYSTEM_PROMPT = (
    "You generate a single, succinct line that situates a chunk within its full document "
    "to improve search retrieval. Output must be strictly: [succinct context] : [original chunk or corrected version]. "
    "If the chunk contains an incomplete number, percentage, or entity, correct it using the full document. "
    "If a sentence is cut off, reconstruct only what is necessary for clarity. "
    "If the chunk is part of a table, include the complete table entry. "
    "Do not add any explanations beyond the required output."
)

ALLOWED_NAMESPACES = [
    "general_hotel_information",
    "room_services",
    "hotel_policies",
    "local_hotel_information",
    "hotel_facilities",
]

# Keep last 5 turns (10 messages) in session history
SESSION_RUN_CONFIG = RunConfig(
    session_input_callback=lambda history_items, new_items: (history_items + new_items)[
        -10:
    ]
)


def build_situated_chunk_instructions(document_text: str, chunk: str) -> str:
    doc = document_text.strip()
    piece = chunk.strip()
    return (
        "<document>\n"
        f"{doc}\n</document>\n\n"
        "Here is the chunk we want to situate within the overall document:\n\n"
        "<chunk>\n"
        f"{piece}\n</chunk>\n\n"
        "Please:\n"
        "- Provide a short and succinct context to situate this chunk within the document for improved search retrieval.\n"
        "- Return the original chunk exactly as provided unless a correction is necessary.\n"
        "- If the chunk contains an incomplete number, percentage, or entity, correct it using the full document.\n"
        "- If part of a sentence is cut off, reconstruct the missing words only if necessary for clarity.\n"
        "- If the chunk is part of a table, include the complete table entry to maintain data integrity\n"
        "- Do not add any additional explanations or formatting beyond the required output.\n\n"
        "Fill in the following format:\n"
        "[succinct context] : [original chunk or corrected version if necessary]"
    )


def build_answer_instructions(context: str) -> str:
    return (
        "Use the following context (delimited by <ctx></ctx>) and the chat history to answer the user query.\n"
        f"<ctx>\n{context}\n</ctx>"
    )
