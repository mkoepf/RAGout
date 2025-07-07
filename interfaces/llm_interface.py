from typing import Protocol


class LLMInterface(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using an LLM."""
        ...
