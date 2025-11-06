# ===========================================
# llm_loader.py (Final Render-Proof Fix)
# ===========================================
import os
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain_community.llms import Ollama

# ✅ Forcefully remove proxy variables BEFORE any Groq import
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    if proxy_var in os.environ:
        print(f"⚙️ Removing proxy var: {proxy_var}")
        os.environ.pop(proxy_var, None)

# Now import Groq safely
try:
    from groq import Groq
except ImportError:
    Groq = None

# Load .env locally (Render ignores)
load_dotenv()


class ChatGroq(LLM):
    """
    LangChain-compatible wrapper for Groq API.
    This version safely handles Render’s proxy injection.
    """

    def __init__(
        self,
        model: str,
        groq_api_key: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        if Groq is None:
            raise ImportError("Groq library not installed. Run `pip install groq`.")

        # Initialize Groq client
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
            return f"❌ Groq API call failed: {e}"

    @property
    def _llm_type(self) -> str:
        return "groq"


def load_llm(default_model="llama3-8b-8192"):
    """
    Load either Groq Cloud or Local Ollama model dynamically.
    """
    backend = os.getenv("LLM_BACKEND", "groq").lower()
    model_name = os.getenv("LLM_MODEL", default_model)

    if backend == "ollama":
        print(f"🧠 Using Local Ollama model: {model_name}")
        return Ollama(model=model_name)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY not found in environment or .env file.")

    print(f"⚡ Using Groq Cloud model: {model_name}")
    return ChatGroq(
        model=model_name,
        groq_api_key=groq_api_key,
        temperature=0.2,
        max_tokens=2048,
    )
