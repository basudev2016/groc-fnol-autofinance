import json
import os

def sop_lookup_tool(user_query: str) -> str:
    """
    Searches SOP data for matching question patterns with error handling.
    """
    sop_path = "data/sop_data.json"

    # If file is missing
    if not os.path.exists(sop_path):
        return "⚠️ SOP data file not found. Please ensure data/sop_data.json exists."

    try:
        with open(sop_path, "r") as f:
            content = f.read().strip()
            if not content:
                return "⚠️ SOP data file is empty. Please repopulate it with valid content."
            sop_data = json.loads(content)
    except json.JSONDecodeError as e:
        return f"⚠️ SOP data file is invalid JSON (error at line {e.lineno}). Please fix and retry."

    query = user_query.lower()

    # Pattern-based matching
    for sop in sop_data:
        for pattern in sop.get("question_patterns", []):
            if pattern in query:
                return f"📘 {sop['response']}"

    # Default fallback if nothing matches
    return "Sorry, I couldn't find a standard process for that. Would you like to connect to an agent?"
