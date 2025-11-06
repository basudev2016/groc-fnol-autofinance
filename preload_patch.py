# ===========================================
# preload_patch.py
# ===========================================
import os

# 🚫 Render injects proxy vars — remove before any imports
for proxy_var in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    if proxy_var in os.environ:
        print(f"⚙️ Removing proxy var: {proxy_var}")
        os.environ.pop(proxy_var, None)
