"""Gradio chat UI for the copilot.

Needs the `ui` extra:  pip install ".[ui]"
Run:                   python scripts/ui.py     -> http://127.0.0.1:7861
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from platform_copilot.dependencies import get_pipeline
from platform_copilot.services.rag.render import render_answer


def respond(message: str, history: list[Any]) -> str:
    return render_answer(get_pipeline().answer(message))


def build_ui() -> gr.ChatInterface:
    """Kept separate from launch() so the UI can be constructed in a test."""
    return gr.ChatInterface(
        fn=respond,
        title="Platform Copilot",
        description=(
            "Ask about the platform's runbooks, ADRs and incidents. Answers are grounded "
            "in the indexed corpus and cite their sources."
        ),
        examples=[
            "What are the first steps for a demand-forecaster drift alert?",
            "How do I escalate if the promotion gate keeps rejecting candidates?",
        ],
    )


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7861)
