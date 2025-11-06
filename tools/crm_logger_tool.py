from datetime import datetime

def crm_logger_tool(user_query: str, agent_response: str) -> str:
    """
    Logs each chat exchange to a local log file.
    """
    log_entry = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
        f"User: {user_query}\n"
        f"Agent: {agent_response}\n\n"
    )

    with open("logs/chat_history.log", "a", encoding="utf-8") as f:
        f.write(log_entry)

    return "🗂️ Interaction logged successfully."
