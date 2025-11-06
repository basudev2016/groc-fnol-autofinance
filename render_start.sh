#!/usr/bin/env bash
# ===========================================
# render_start.sh — final Render-safe startup
# ===========================================

echo "🧹 Removing proxy environment variables..."
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset https_proxy
unset all_proxy

echo "🚀 Starting Streamlit app..."
streamlit run app.py --server.port $PORT --server.headless true
