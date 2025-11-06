# import re
# import pandas as pd
# from tools.verify_user_tool import verify_user_tool
# from tools.payment_lookup_tool import payment_lookup_tool
# from tools.fnol_claim_tool import fnol_claim_tool
# from tools.sop_lookup_tool import sop_lookup_tool
# from tools.crm_logger_tool import crm_logger_tool


# class AutoFinanceAgent:
#     def __init__(self):
#         self.verified_customer = None
#         self.last_greeted = False

#     def route_query(self, user_input: str) -> str:
#         text = user_input.lower().strip()

#         # -----------------------------
#         # STEP 0: GREETING / SMALL TALK
#         # -----------------------------
#         if any(
#             phrase in text
#             for phrase in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
#         ):
#             self.last_greeted = True
#             response = (
#                 "👋 Hello! Welcome to Auto Finance Support.\n\n"
#                 "To help you better, please verify yourself.\n"
#                 "You can say things like:\n"
#                 "- 'I am John Doe'\n"
#                 "- 'My loan number is LN001'\n"
#                 "- 'My phone number is 9876543210'"
#             )
#             crm_logger_tool(user_input, response)
#             return response

#         # -----------------------------
#         # STEP 1: CUSTOMER VERIFICATION
#         # -----------------------------
#         if any(
#             phrase in text
#             for phrase in [
#                 "verify",
#                 "customer id",
#                 "loan number",
#                 "name",
#                 "identify",
#                 "who am i",
#                 "i am",
#                 "my name is",
#                 "this is",
#                 "phone number",
#             ]
#         ):
#             response = verify_user_tool(user_input)
#             if "✅ Verified customer" in response:
#                 self.verified_customer = self.extract_customer_id(response)
#             crm_logger_tool(user_input, response)
#             return response

#         # -----------------------------
#         # STEP 2: PAYMENT INFORMATION
#         # -----------------------------
#         if any(
#             phrase in text
#             for phrase in [
#                 "payment",
#                 "emi",
#                 "due date",
#                 "balance",
#                 "outstanding",
#                 "loan amount",
#                 "installment",
#             ]
#         ):
#             if not self.verified_customer:
#                 response = (
#                     "Before I can access your payment details, please verify yourself.\n"
#                     "Can you share your name and either your loan number or phone number?"
#                 )
#             else:
#                 response = payment_lookup_tool(self.verified_customer)
#             crm_logger_tool(user_input, response)
#             return response

#         # -----------------------------
#         # STEP 3: FNOL / CLAIM PROCESS
#         # -----------------------------
#         if any(
#             phrase in text
#             for phrase in [
#                 "accident",
#                 "damage",
#                 "theft",
#                 "claim",
#                 "fnol",
#                 "file claim",
#                 "report loss",
#             ]
#         ):
#             if not self.verified_customer:
#                 response = (
#                     "Please verify your identity first before we log your claim.\n"
#                     "You can say 'I am John Doe' or provide your loan number."
#                 )
#             else:
#                 response = fnol_claim_tool(self.verified_customer)
#             crm_logger_tool(user_input, response)
#             return response

#         # -----------------------------
#         # STEP 4: SOP / PROCESS HELP
#         # -----------------------------
#         sop_response = sop_lookup_tool(user_input)
#         if "couldn't find a standard process" in sop_response.lower():
#             # If user has just greeted but not verified, prompt politely
#             if not self.verified_customer:
#                 sop_response = (
#                     "I'm here to help with auto finance services like payments, claims, and insurance.\n"
#                     "Please tell me your name or loan number to begin verification."
#                 )
#         crm_logger_tool(user_input, sop_response)
#         return sop_response

#     # -----------------------------
#     # HELPER FUNCTION
#     # -----------------------------
#     def extract_customer_id(self, response_text: str) -> str:
#         match = re.search(r"\(Loan: (LN\d+)", response_text)
#         if match:
#             loan_number = match.group(1)
#             df = pd.read_csv("data/customers.csv")
#             cust = df[df["loan_number"] == loan_number]
#             if not cust.empty:
#                 return cust.iloc[0]["customer_id"]
#         return None

import re
import pandas as pd
from tools.verify_user_tool import verify_user_tool
from tools.payment_lookup_tool import payment_lookup_tool
from tools.fnol_claim_tool import fnol_claim_tool
from tools.sop_lookup_tool import sop_lookup_tool
from tools.crm_logger_tool import crm_logger_tool


class AutoFinanceAgent:
    def __init__(self):
        self.verified_customer = None
        self.last_greeted = False

    def route_query(self, user_input: str) -> str:
        text = user_input.lower().strip()

        # -----------------------------
        # STEP 0: GREETING / SMALL TALK
        # -----------------------------
        if any(
            phrase in text
            for phrase in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        ):
            self.last_greeted = True
            response = (
                "👋 Hello! Welcome to Auto Finance Support.\n\n"
                "To help you better, please verify yourself.\n"
                "You can say things like:\n"
                "- 'I am John Doe'\n"
                "- 'My loan number is LN001'\n"
                "- 'My phone number is 9876543210'"
            )
            crm_logger_tool(user_input, response)
            return response

        # -----------------------------
        # STEP 1: CUSTOMER VERIFICATION
        # -----------------------------
        if any(
            phrase in text
            for phrase in [
                "verify",
                "customer id",
                "loan number",
                "name",
                "identify",
                "who am i",
                "i am",
                "my name is",
                "this is",
                "phone number",
            ]
        ):
            response = verify_user_tool(user_input)
            if "✅ Verified customer" in response:
                self.verified_customer = self.extract_customer_id(response)
            crm_logger_tool(user_input, response)
            return response

        # -----------------------------
        # STEP 2: PAYMENT INFORMATION
        # -----------------------------
        if any(
            phrase in text
            for phrase in [
                "payment",
                "emi",
                "due date",
                "balance",
                "outstanding",
                "loan amount",
                "installment",
            ]
        ):
            if not self.verified_customer:
                response = (
                    "Before I can access your payment details, please verify yourself.\n"
                    "Can you share your name and either your loan number or phone number?"
                )
            else:
                response = payment_lookup_tool(self.verified_customer)
            crm_logger_tool(user_input, response)
            return response

        # -----------------------------
        # STEP 3: FNOL / CLAIM PROCESS
        # -----------------------------
        if any(
            phrase in text
            for phrase in [
                "accident",
                "damage",
                "theft",
                "claim",
                "fnol",
                "file claim",
                "report loss",
            ]
        ):
            if not self.verified_customer:
                response = (
                    "Please verify your identity first before we log your claim.\n"
                    "You can say 'I am John Doe' or provide your loan number."
                )
            else:
                response = fnol_claim_tool(self.verified_customer)
            crm_logger_tool(user_input, response)
            return response

        # -----------------------------
        # STEP 4: SESSION SUMMARY (SMART END)
        # -----------------------------
        if any(
            phrase in text
            for phrase in ["thanks", "thank you", "goodbye", "end chat", "that's all", "bye"]
        ):
            summary = self.generate_summary()
            response = f"🧾 Here's a quick summary of our chat today:\n\n{summary}\n\nThank you for connecting with Auto Finance Support!"
            crm_logger_tool(user_input, response)
            return response

        # -----------------------------
        # STEP 5: SOP / PROCESS HELP
        # -----------------------------
        sop_response = sop_lookup_tool(user_input)
        if "couldn't find a standard process" in sop_response.lower():
            if not self.verified_customer:
                sop_response = (
                    "I'm here to help with auto finance services like payments, claims, and insurance.\n"
                    "Please tell me your name or loan number to begin verification."
                )
        crm_logger_tool(user_input, sop_response)
        return sop_response

    # -----------------------------
    # HELPER FUNCTIONS
    # -----------------------------
    def extract_customer_id(self, response_text: str) -> str:
        match = re.search(r"\(Loan: (LN\d+)", response_text)
        if match:
            loan_number = match.group(1)
            df = pd.read_csv("data/customers.csv")
            cust = df[df["loan_number"] == loan_number]
            if not cust.empty:
                return cust.iloc[0]["customer_id"]
        return None

    def generate_summary(self) -> str:
        """
        Reads the chat log and summarizes key actions.
        """
        try:
            with open("logs/chat_history.log", "r", encoding="utf-8") as f:
                text = f.read().lower()
        except FileNotFoundError:
            return "No chat history found for this session."

        summary_points = []
        if "verified customer" in text:
            summary_points.append("✅ You were successfully verified.")
        if "loan number" in text or "emi" in text or "payment" in text:
            summary_points.append("💰 We discussed your payment or EMI details.")
        if "claim" in text:
            summary_points.append("🧾 We filed or checked an insurance claim.")
        if "insurance" in text or "coverage" in text:
            summary_points.append("📘 We reviewed insurance coverage or SOP details.")

        if not summary_points:
            return "No significant actions detected."
        return "\n".join(summary_points)
