# ===========================================
# llm_loader.py (Final Stable Version)
# ===========================================
import os
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain_community.llms import Ollama
from groq import Groq

load_dotenv()


class ChatGroq(LLM):
    """
    LangChain-compatible wrapper for the Groq API.
    Supports callbacks, metadata, cache, and is fully Pydantic-safe.
    """

    def __init__(
        self,
        model: str,
        groq_api_key: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        callbacks=None,
        tags=None,
        metadata=None,
        verbose=False,
        cache=None,
    ):
        # Use object.__setattr__ to bypass Pydantic's BaseModel field restriction
        object.__setattr__(self, "client", Groq(api_key=groq_api_key))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_tokens", max_tokens)

        # Required attributes for LangChain LLM interface
        object.__setattr__(self, "callbacks", callbacks or [])
        object.__setattr__(self, "tags", tags or [])
        object.__setattr__(self, "metadata", metadata or {})
        object.__setattr__(self, "verbose", verbose)
        object.__setattr__(self, "cache", cache)  # ✅ NEW — fixes AttributeError

    def _call(self, prompt: str, **kwargs) -> str:
        """
        Execute a Groq chat completion and return model output.
        """
        completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"


def load_llm(default_model="llama3-8b-8192"):
    """
    Dynamically load either Groq Cloud LLM or Local Ollama
    based on environment variable LLM_BACKEND.
    """
    backend = os.getenv("LLM_BACKEND", "groq").lower()
    model_name = os.getenv("LLM_MODEL", default_model)

    if backend == "ollama":
        print(f"🧠 Using Local Ollama model: {model_name}")
        return Ollama(model=model_name)
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file.")
        print(f"⚡ Using Groq Cloud model: {model_name}")
        return ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key,
            temperature=0.2,
            max_tokens=2048,
        )
