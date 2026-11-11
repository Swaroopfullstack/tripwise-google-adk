"""Authenticated local Foundation Studio with incremental responses."""
import os

import gradio as gr

from .config import Settings
from .foundations import stream_reply


def create_demo():
    settings = Settings()
    return gr.ChatInterface(fn=stream_reply, title="Foundation Studio",
            description=f"Mode: {settings.run_mode}. Learn prompts, runnables and checkpointing. No shell execution.",
            examples=["Explain a reducer using a short example", "Why does an agent need a checkpointer?"],
            api_visibility="private")


def main():
    settings = Settings()
    credential = os.getenv("GRADIO_PASSWORD", "")
    if settings.app_env == "production" and len(credential) < 24:
        raise ValueError("Set GRADIO_PASSWORD to at least 24 characters for production")
    create_demo().launch(server_name="127.0.0.1", server_port=7860, share=False,
                        auth=("portfolio", credential) if credential else None, max_threads=4)


if __name__ == "__main__":
    main()
