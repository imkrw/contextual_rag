import functools
import logging
import os
import asyncio
import retrieval.models as models
import gradio as gr
from typing import List, Optional
from agents import Agent, Runner, SQLiteSession
from openai import OpenAI
from core.session import ChatSession, reset_session
from core.embeddings import get_embedding
from retrieval.index import get_pinecone_index
from ingest.uploader import ALLOWED_NAMESPACES, uploader
from core.presets import (
    ENHANCER_PROMPTS,
    INSTRUCTION_CLASSIFIER,
    SESSION_RUN_CONFIG,
    build_answer_instructions,
)
from core.utils import build_context, ensure_environment_ready
from retrieval.query import query_pinecone


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _enhance_query(intent: str, user_text: str):
    enhancer_instructions = ENHANCER_PROMPTS.get(intent)
    if enhancer_instructions:
        enhancer = Agent(name="Enhancer", instructions=enhancer_instructions)
        enhance_response = await Runner.run(enhancer, user_text)
        logger.info(
            f"{enhance_response.context_wrapper.usage.total_tokens} tokens used for enhancement"
        )
        return (enhance_response.final_output or user_text).strip()
    return user_text


async def _get_embedding_task(client: OpenAI, user_text: str):
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_embedding, client, user_text)
    except Exception:
        return None


async def _handle_message(
    user_message: str,
    chat_history: Optional[List[dict]],
    chat_session: ChatSession,
    client: OpenAI,
):
    session = chat_session.ensure()
    response = await _run_turn(session, user_message, client)
    history = chat_history or []
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]
    return updated_history, ""


async def _run_turn(session: SQLiteSession, user_text: str, client: OpenAI) -> str:
    # Classification
    classifier = Agent(name="Classifier", instructions=INSTRUCTION_CLASSIFIER)
    classify_res = await Runner.run(classifier, user_text)
    intent = (
        (classify_res.final_output or "other_unrelated")
        .strip()
        .lower()
        .replace("-", "_")
    )

    logger.info(f"Classified intent: {intent}")
    logger.info(
        f"{classify_res.context_wrapper.usage.total_tokens} tokens used for classification"
    )

    if intent == "other_unrelated":
        other_unrelated = Agent(name="other_unrelated")
        final = await Runner.run(
            other_unrelated,
            user_text,
            session=session,
            run_config=SESSION_RUN_CONFIG,
        )
        logger.info(
            f"{final.context_wrapper.usage.total_tokens} tokens used for other_unrelated"
        )
        return (final.final_output or "Something went wrong, please try again.").strip()

    # Parallel
    enhanced_query, embedding = await asyncio.gather(
        _enhance_query(intent, user_text), _get_embedding_task(client, user_text)
    )

    index = get_pinecone_index()
    results: List[models.Match] = []
    if index and embedding:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, query_pinecone, index, embedding, intent, 5
        )

    concatenated = build_context(results)
    instructions = build_answer_instructions(concatenated)
    answer_agent = Agent(name="Answer", instructions=instructions)
    final = await Runner.run(
        answer_agent,
        enhanced_query,
        session=session,
        run_config=SESSION_RUN_CONFIG,
    )
    logger.info(
        f"{final.context_wrapper.usage.total_tokens} tokens used for final answer"
    )
    return final.final_output or "Something went wrong, please try again."


def _build_gradio_app(client: OpenAI, session_db: str):
    chat_session = ChatSession(session_db)
    handle = functools.partial(
        _handle_message, chat_session=chat_session, client=client
    )
    reset = functools.partial(reset_session, chat_session=chat_session)
    upload = functools.partial(uploader, client=client)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=7):
                gr.Markdown("### Contextual retrieval augmented generation.")
                chatbot = gr.Chatbot(type="messages", label="Conversation")
                message_box = gr.Textbox(
                    label="Query", placeholder="Type your message..."
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    reset_button = gr.Button("Reset", variant="secondary")

            with gr.Column(scale=3):
                gr.Markdown("### Document uploader")
                namespace_input = gr.Dropdown(
                    label="Namespace",
                    choices=ALLOWED_NAMESPACES,
                    multiselect=False,
                )
                file_input = gr.File(
                    label="Upload .txt files",
                    file_count="multiple",
                    file_types=[".txt"],
                )
                upload_button = gr.Button("Upload", variant="primary")
                upload_output = gr.Textbox(
                    label="Upload log",
                    show_copy_button=True,
                    lines=10,
                )

        message_box.submit(
            handle,
            inputs=[message_box, chatbot],
            outputs=[chatbot, message_box],
        )
        send_button.click(
            handle,
            inputs=[message_box, chatbot],
            outputs=[chatbot, message_box],
        )
        reset_button.click(
            reset,
            outputs=[chatbot, message_box],
        )
        upload_button.click(
            upload,
            inputs=[namespace_input, file_input],
            outputs=[upload_output],
        )

    return demo


def _build_app() -> gr.Blocks:
    ensure_environment_ready()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    session_db = "conversation.db"
    return _build_gradio_app(client, session_db)


demo = _build_app()


if __name__ == "__main__":
    demo.launch()
