
import streamlit as st
import os
import pandas as pd
import preload_patch  # 🩵 fixes Render proxy issue before any import
# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(page_title="Auto Finance GROC Assistant", layout="centered")

st.title("🚗 Auto Finance Agentic Assistant")
st.caption("Run locally — with Rule-based, LLM, GROC, or Multi-Agent Supervisor intelligence.")

# ===========================
# SIDEBAR CONTROLS
# ===========================
st.sidebar.header("⚙️ Agent Mode Selector")

mode = st.sidebar.selectbox(
    "Select Agent Mode",
    [
        "Supervisor (Multi-Agent)",
        "GROC (Planner + Executor)",
        "Local LLM (Ollama)",
        "Rule-Based"
    ],
)

if mode == "Supervisor (Multi-Agent)":
    from supervisor_agent import SupervisorAgent as AutoFinanceAgent
elif mode == "GROC (Planner + Executor)":
    from agent_groc import AutoFinanceGROC as AutoFinanceAgent
elif mode == "Local LLM (Ollama)":
    from agent_logic_llm import AutoFinanceLLMAgent as AutoFinanceAgent
else:
    from agent_logic import AutoFinanceAgent

# ===========================
# INITIALIZE AGENT & STATE
# ===========================
if "agent" not in st.session_state:
    st.session_state.agent = AutoFinanceAgent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = st.session_state.agent

# ===========================
# SIDEBAR INFO
# ===========================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Agent Overview")
st.sidebar.info(
    """
**Available Modes**
- 🧠 Supervisor: Multi-agent planner + reflector  
- 🧭 GROC: Goal-oriented reasoning chain  
- 🤖 Local LLM: Direct reasoning (Ollama)  
- ⚙️ Rule-Based: Baseline offline logic  
"""
)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.agent = AutoFinanceAgent()
    st.success("Chat cleared and agent reinitialized.")

# ===========================
# DISPLAY VERIFIED BADGE (if applicable)
# ===========================
if hasattr(agent, "verified_customer") and agent.verified_customer:
    try:
        df = pd.read_csv("data/customers.csv")
        row = df[df["customer_id"] == agent.verified_customer].iloc[0]
        name = f"{row['first_name']} {row['last_name']}"
        st.markdown(f"### 🟢 Verified: **{name}**")
    except Exception:
        st.markdown("### 🟢 Verified Customer")

# ===========================
# CHAT INTERFACE
# ===========================
st.markdown("### 💬 Chat with the Assistant")
user_input = st.chat_input("Type your message or goal...")

if user_input:
    # Choose appropriate method for each mode
    if mode == "Supervisor (Multi-Agent)":
        response = agent.orchestrate_goal(user_input)
    elif mode == "GROC (Planner + Executor)":
        response = agent.handle_goal(user_input)
    else:
        # Both LLM and Rule-Based have route_query
        response = agent.route_query(user_input)

    # Append to chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("agent", response))

# ===========================
# DISPLAY CHAT HISTORY
# ===========================
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)

# ===========================
# FOOTER INFO
# ===========================
st.divider()
st.markdown(
    """
📜 **Log file:** `logs/chat_history.log`  
💾 All chats, plans, and reflections are stored for CRM reference.  
🧠 Try complex goals like:
- "Help me check my EMI and file a claim."
- "Explain my insurance coverage and next EMI."
"""
)
