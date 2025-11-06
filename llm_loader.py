# ===========================================
# llm_loader.py (Final Stable + Render Safe)
# ===========================================
import os
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain_community.llms import Ollama

# Try importing Groq (optional dependency)
try:
    from groq import Groq
except ImportError:
    Groq = None

# Load .env only for local testing — Render ignores this automatically
load_dotenv()


class ChatGroq(LLM):
    """
    LangChain-compatible wrapper for Groq API.
    Render-safe version that avoids proxy issues.
    """

    def __init__(
        self,
        model: str,
        groq_api_key: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        if Groq is None:
            raise ImportError(
                "Groq library not installed. Please install it using `pip install groq`."
            )

        # ✅ Prevent Render proxy injection conflicts
        for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            if proxy_var in os.environ:
                print(f"⚙️ Removing proxy variable: {proxy_var}")
                del os.environ[proxy_var]

        # ✅ Initialize Groq client safely
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, prompt: str, **kwargs) -> str:
        """
        Execute a Groq chat completion and return model output.
        """
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"❌ Groq API call failed: {e}")
            return f"[Error: Unable to reach Groq API - {e}]"

    @property
    def _llm_type(self) -> str:
        return "groq"


def load_llm(default_model="llama3-8b-8192"):
    """
    Dynamically load LLM backend — Groq Cloud (default) or Ollama local.
    """
    backend = os.getenv("LLM_BACKEND", "groq").lower()
    model_name = os.getenv("LLM_MODEL", default_model)

    # ✅ Case 1: Ollama (Local)
    if backend == "ollama":
        print(f"🧠 Using Local Ollama model: {model_name}")
        return Ollama(model=model_name)

    # ✅ Case 2: Groq Cloud (Default)
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "❌ GROQ_API_KEY not found in environment or .env file. "
            "Set it in Render > Environment tab or in your .env locally."
        )

    print(f"⚡ Using Groq Cloud model: {model_name}")
    return ChatGroq(
        model=model_name,
        groq_api_key=groq_api_key,
        temperature=0.2,
        max_tokens=2048,
    )
