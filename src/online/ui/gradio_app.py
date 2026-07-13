import requests
import gradio as gr
from loguru import logger
from src.shared.config import settings
from src.personal_ai_engine.domain.queries import SearchQuery
from src.personal_ai_engine.domain.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from src.online.retrieval.retriever import DocumentRetriever
from src.offline.mongo_client import MongoDBClient
from src.gateway.factory import AIGatewayFactory


def check_infrastructure_status() -> dict[str, str]:
    status = {}

    # 1. MongoDB check
    try:
        with MongoDBClient() as client:
            if client.ping():
                status["mongodb"] = "🟢 Online"
            else:
                status["mongodb"] = "🔴 Offline"
    except Exception:
        status["mongodb"] = "🔴 Offline (Connection Failed)"

    # 2. Provider check
    provider = settings.AI_PROVIDER
    if provider == "ollama":
        try:
            res = requests.get(settings.OLLAMA_BASE_URL, timeout=2)
            if res.status_code == 200:
                status["provider"] = f"🟢 Ollama Online ({settings.OLLAMA_CHAT_MODEL})"
            else:
                status["provider"] = f"🔴 Ollama Offline (Status {res.status_code})"
        except Exception:
            status["provider"] = "🔴 Ollama Offline (Refused)"
    elif provider == "gemini":
        if settings.GEMINI_API_KEY:
            status["provider"] = (
                f"🟡 Gemini ({settings.GEMINI_CHAT_MODEL} - API Key Present)"
            )
        else:
            status["provider"] = "🔴 Gemini API Key Missing"
    elif provider == "openrouter":
        if settings.OPENROUTER_API_KEY:
            status["provider"] = (
                f"🟡 OpenRouter ({settings.OPENROUTER_CHAT_MODEL} - API Key Present)"
            )
        else:
            status["provider"] = "🔴 OpenRouter API Key Missing"

    return status


def answer_question(question: str, strategy: str, history: list[list[str]]):
    if not question.strip():
        return history, "", "No query entered.", "No source metadata available."

    # Perform infrastructure checks
    infra = check_infrastructure_status()
    if "🔴" in infra["mongodb"]:
        err_msg = "Database Error: MongoDB is offline. Please make sure Docker services are running."
        history.append([question, err_msg])
        return history, "", err_msg, ""

    if "🔴" in infra["provider"] or "API Key Missing" in infra["provider"]:
        err_msg = f"Provider Error: The active provider '{settings.AI_PROVIDER}' is unavailable or unconfigured."
        history.append([question, err_msg])
        return history, "", err_msg, ""

    try:
        # 1. Retrieve relevant chunks
        retriever = DocumentRetriever(strategy=strategy.lower())
        results = retriever.retrieve(
            SearchQuery(text=question, top_k=settings.RETRIEVAL_TOP_K)
        )

        if not results:
            context_str = "No relevant context found."
            sources_html = "<div style='color: orange;'>No relevant sources found in the database.</div>"
            metadata_str = "None"
        else:
            # Format sources for UI display
            sources_list = []
            metadata_list = []
            context_blocks = []

            for idx, (chunk, score) in enumerate(results):
                title = chunk.metadata.get("title", "Untitled")
                url = chunk.metadata.get("url", "#")
                snippet = chunk.content[:200] + "..."

                # HTML display card for sources
                sources_list.append(
                    f"<div style='border: 1px solid #ddd; padding: 10px; margin-bottom: 8px; border-radius: 6px; background-color: #f9f9f9;'>"
                    f"<strong>[{idx + 1}] <a href='{url}' target='_blank' style='color: #0066cc; text-decoration: none;'>{title}</a></strong><br/>"
                    f"<span style='font-size: 0.85em; color: #666;'>Relevance Score: {score:.4f} | Link: <code style='font-size: 0.95em;'>{url}</code></span><br/>"
                    f"<p style='margin: 5px 0 0 0; font-size: 0.9em; font-style: italic;'>{snippet}</p>"
                    f"</div>"
                )

                context_blocks.append(
                    f"Source: {title} ({url})\nContent:\n{chunk.content}"
                )
                metadata_list.append(chunk.model_dump())

            context_str = "\n\n".join(context_blocks)
            sources_html = "".join(sources_list)
            import json

            metadata_str = json.dumps(metadata_list, indent=2)

        # 2. Generate response using LLM Gateway
        chat_generator = AIGatewayFactory.get_chat_generator()
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)
        user_prompt = RAG_USER_PROMPT.format(question=question)

        answer = chat_generator.generate_with_system(
            system=system_prompt, prompt=user_prompt
        )
        history.append([question, answer])

        return history, "", sources_html, metadata_str

    except Exception as e:
        logger.exception(f"Error answering question: {e}")
        err_msg = f"Runtime Exception: {e}"
        history.append([question, err_msg])
        return history, "", err_msg, ""


def build_app() -> gr.Blocks:
    status = check_infrastructure_status()

    with gr.Blocks(title="Personal AI Engine") as demo:
        gr.Markdown(
            "# 🧠 Personal AI Engine\n"
            "Ask questions about your ingested Notion database pages and crawled websites. "
            "Leverage semantic, keyword, or hybrid reciprocal rank fusion (RRF) retrieval strategies."
        )

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Conversation")
                msg = gr.Textbox(
                    placeholder="Enter your question here...",
                    label="Ask a Question",
                    show_label=False,
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit Query", variant="primary")
                    clear_btn = gr.Button("Clear Chat History")

            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Engine Settings")
                strategy = gr.Radio(
                    choices=["Hybrid", "Semantic", "Keyword"],
                    value="Hybrid",
                    label="Search Strategy",
                )

                gr.Markdown("### 📶 Infrastructure Status")
                db_status = gr.Textbox(
                    value=status.get("mongodb", "Unknown"),
                    label="MongoDB Connection",
                    interactive=False,
                )
                provider_status = gr.Textbox(
                    value=status.get("provider", "Unknown"),
                    label="Model Provider",
                    interactive=False,
                )
                refresh_btn = gr.Button("Refresh Status", size="sm")

        with gr.Row():
            with gr.Tab("Retrieved Sources"):
                sources_display = gr.HTML(
                    value="<div style='color: gray;'>Submit a query to inspect retrieved documents.</div>",
                    label="Sources Cards",
                )
            with gr.Tab("Raw Chunks Metadata"):
                metadata_display = gr.Code(
                    value="No chunk metadata available.",
                    language="json",
                    label="Raw Chunks JSON",
                )

        # Refresh button callback
        def refresh_ui_status():
            cur = check_infrastructure_status()
            return cur["mongodb"], cur["provider"]

        refresh_btn.click(refresh_ui_status, outputs=[db_status, provider_status])

        # Submit callbacks
        submit_btn.click(
            answer_question,
            inputs=[msg, strategy, chatbot],
            outputs=[chatbot, msg, sources_display, metadata_display],
        )
        msg.submit(
            answer_question,
            inputs=[msg, strategy, chatbot],
            outputs=[chatbot, msg, sources_display, metadata_display],
        )

        # Clear callback
        clear_btn.click(
            lambda: (
                [],
                "",
                "<div style='color: gray;'>Submit a query to inspect retrieved documents.</div>",
                "No chunk metadata available.",
            ),
            outputs=[chatbot, msg, sources_display, metadata_display],
        )

    return demo


def launch_app():
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.GRADIO_PORT,
        share=settings.GRADIO_SHARE,
    )


if __name__ == "__main__":
    launch_app()
