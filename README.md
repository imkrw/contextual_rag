## contextual-rag

Contextual RAG assistant powered by OpenAI, Pinecone, and Gradio.
 
- `uv sync`
- Export credentials:
   ```bash
   export OPENAI_API_KEY=your_key
   export PINECONE_API_KEY=your_key
   export PINECONE_INDEX=your_key
   ```
- `uv run python3 main.py`

Upload `.txt` files in the UI to add context and ask hotel-related question or other-unrelated questions in the chat panel.

<image src="media/demo.png">
