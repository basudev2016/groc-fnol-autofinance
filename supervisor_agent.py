# ===========================================
# supervisor_agent.py (Final Interactive Version)
# ===========================================

import re
from agent_groc import AutoFinanceGROC
from tools.crm_logger_tool import crm_logger_tool
from tools.verify_user_tool import verify_user_tool
from llm_loader import load_llm


class SupervisorAgent:
    """
    The Supervisor Agent coordinates all sub-agents and ensures proper orchestration:
    - Verifies user identity (via verify_user_tool)
    - Routes tasks to GROC Agent (payments, claims, SOP)
    - Reflects on task success
    """

    def __init__(self, model_name="mistral"):
        # ✅ Load the Groq/Ollama LLM dynamically
        self.llm = load_llm()
        self.groc_agent = AutoFinanceGROC(model_name)
        self.completed_steps = []
        self.user_context = None  # Store verified user details for the session

    def orchestrate_goal(self, user_goal: str):
        print(f"🎯 Received Goal: {user_goal}")

        # ------------------------------------------------------------
        # 1️⃣ Step 1: Verification (only if not yet verified)
        # ------------------------------------------------------------
        if not self.user_context or not self.user_context.get("ok"):
            query = {}

            # Extract LoanID (like LN001)
            loan_match = re.search(r"\bLN\d+\b", user_goal, re.IGNORECASE)
            if loan_match:
                query["loan"] = loan_match.group(0)

            # Extract 10-digit phone number
            phone_match = re.search(r"\b\d{10}\b", user_goal)
            if phone_match:
                query["phone"] = phone_match.group(0)

            # Try name if present
            name_match = re.search(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", user_goal)
            if name_match:
                query["name"] = name_match.group(0)

            verification_result = verify_user_tool(query)
            if verification_result.get("ok"):
                self.user_context = verification_result
                verified_name = verification_result.get("CustomerName", "Customer")
                verified_loan = verification_result.get("LoanID", "N/A")
                print(f"✅ Verified user: {verified_name} (Loan: {verified_loan})")

                # 👇 NEW LOGIC: Don't route “LN001” to GROC immediately
                # Ask user what they want next
                if (
                    "emi" not in user_goal.lower()
                    and "claim" not in user_goal.lower()
                    and "insurance" not in user_goal.lower()
                    and "coverage" not in user_goal.lower()
                    and "payment" not in user_goal.lower()
                ):
                    return (
                        f"✅ Verified {verified_name} (Loan {verified_loan}).\n"
                        "What would you like to do next — check your EMI, claim status, or insurance coverage?"
                    )

            else:
                return (
                    "🔐 Verification Agent: Please provide your loan number "
                    "(e.g., LN001) or registered phone number for verification."
                )

        # ------------------------------------------------------------
        # 2️⃣ Step 2: Route Goal to GROC Agent (with verified context)
        # ------------------------------------------------------------
        try:
            result = self.groc_agent.handle_goal(user_goal, context=self.user_context)
            self.completed_steps.append(result)
        except Exception as e:
            result = f"⚠️ GROC Agent Error — {type(e).__name__}: {e}"
            print(result)

        # ------------------------------------------------------------
        # 3️⃣ Step 3: Reflection Phase (concise reasoning)
        # ------------------------------------------------------------
        reflection_prompt = f"""
        You are the Supervisor reviewing the GROC Agent's execution results.

        Goal: {user_goal}
        Result: {result}

        Determine if the user's goal was fully achieved.
        Respond with:
        - "YES" or "NO"
        - Followed by a one-line reasoning.
        """
        try:
            reflection = self.llm.invoke(reflection_prompt)
        except Exception as e:
            reflection = f"⚠️ Reflection step failed — {type(e).__name__}: {e}"

        # ------------------------------------------------------------
        # 4️⃣ Step 4: Log to CRM for traceability
        # ------------------------------------------------------------
        crm_logger_tool(
            user_goal,
            f"Context: {self.user_context}\nResult: {result}\nReflection: {reflection}"
        )

        # ------------------------------------------------------------
        # 5️⃣ Step 5: Return Clean Output
        # ------------------------------------------------------------
        return f"{result}\n\n✅ Reflection: {reflection}"
