"""Resource settings shared by the chat, vision and embedding clients."""

import os


def keep_alive_seconds():
    """Use seconds, supported by both LangChain's chat and embedding clients."""
    value = int(os.getenv("OLLAMA_KEEP_ALIVE_SECONDS", "1800"))
    if value < -1:
        raise ValueError("OLLAMA_KEEP_ALIVE_SECONDS must be -1 or a non-negative integer")
    return value


def configure_cpu_threads(torch):
    """Preserve PyTorch's hardware defaults unless the operator overrides them."""
    value = os.getenv("RAG_CPU_THREADS", "").strip()
    if value:
        threads = int(value)
        if threads < 1:
            raise ValueError("RAG_CPU_THREADS must be a positive integer")
        torch.set_num_threads(threads)
