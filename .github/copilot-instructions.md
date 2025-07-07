# 🧭 Copilot Instructions: LLM Integration in RAGout

## ✅ Goals

- Implement, test, and maintain LLM components within the RAGout system.
- Follow the existing abstraction model (`LLMInterface`) and infrastructure pattern.
- Keep implementations loosely coupled and independently testable.

## ⚙️ Tasks

### General Guidelines
- Always add docstrings to classes and methods.
- Add type annotations wherever possible.

### 🔧 When implementing an LLM backend:
- Create a class in `infrastructure/` that implements `LLMInterface`.
- Ensure it has a `generate(prompt: str) -> str` method.
- Example reference: `OllamaLLM` in `infrastructure/llm_ollama.py`.

### 🔁 When extending or modifying LLM logic:
- Do not hardcode logic outside of the interface boundary.
- Use constructor arguments for configurables (model name, base URL, etc).
- Validate inputs and raise exceptions on errors (e.g., non-200 HTTP status).

### 🧪 When writing tests:
- Add test files to `tests/`, named like `test_llm_*.py`.
- Use mocks to isolate the LLM from external APIs (e.g., `requests.post`).
- Ensure test cases cover successful generation and error handling.

### 💡 When adding a new LLM provider:
- Implement `LLMInterface` in a new file in `infrastructure/`.
- Register it in the composition root if a factory or DI mechanism exists.
- Avoid coupling with retrieval, embedding, or prompt templates.

## 💬 When generating examples:
- Use `OllamaLLM` and prompts that include both context and question.
- Follow this pattern:
  ```python
  from rag_pdf.infrastructure.llm_ollama import OllamaLLM

  llm = OllamaLLM()
  prompt = "What is the meaning of life?"
  answer = llm.generate(prompt)
  print(answer)
  ```

## 🧠 Principles

- Favor composition over inheritance.
- Keep each module single-responsibility.
- Separate I/O, logic, and interfaces.
- Prioritize mocking and testability.

## 📎 Key Files

- `interfaces/llm_interface.py`: Interface definition
- `infrastructure/llm_ollama.py`: Ollama implementation
- `tests/test_llm_ollama.py`: Unit tests
