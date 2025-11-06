# ===========================================
# agent_groc.py (Final Version - Fully Header-Aware & EMI-Fixed)
# ===========================================

import pandas as pd
import os
import json
import hashlib
from tools.crm_logger_tool import crm_logger_tool

# Data folder
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
CUSTOMER_FILE = os.path.join(DATA_PATH, "customers.csv")
CLAIMS_FILE = os.path.join(DATA_PATH, "claims.csv")
PAYMENT_FILE = os.path.join(DATA_PATH, "payments.csv")
SOP_FILE = os.path.join(DATA_PATH, "sop.json")


class AutoFinanceGROC:
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.customers = self._load_csv(CUSTOMER_FILE)
        self.claims = self._load_csv(CLAIMS_FILE)
        self.payments = self._load_csv(PAYMENT_FILE)
        self.sop = self._load_json(SOP_FILE)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _load_csv(self, file_path):
        """Load CSV flexibly and normalize column headers"""
        if not os.path.exists(file_path):
            return pd.DataFrame()

        df = pd.read_csv(file_path, dtype=str).fillna("")

        # Normalize column names (case-insensitive + alias handling)
        rename_map = {}
        for col in df.columns:
            c = col.strip().lower()
            if c == "loan_number":
                rename_map[col] = "LoanID"
            elif c == "customer_id":
                rename_map[col] = "CustomerID"
            elif c == "claim_id":
                rename_map[col] = "ClaimID"
            elif c == "claim_status":
                rename_map[col] = "Status"
            elif c == "settlement_date":
                rename_map[col] = "Settlement_Date"
            elif c == "emi_amount":
                rename_map[col] = "EMI_Amount"
            elif c == "next_due_date":
                rename_map[col] = "Due_Date"
            elif c == "outstanding_balance":
                rename_map[col] = "Balance"
        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _load_json(self, file_path):
        if not os.path.exists(file_path):
            return {
                "coverage_overview": "Comprehensive motor insurance policy.",
                "standard_features": ["Third-party liability", "Own damage"],
                "common_addons": [],
                "claim_process_steps": [
                    "Inform insurer within 24 hours",
                    "Submit RC, DL, photos, FIR (if applicable)",
                    "Surveyor inspection & approval",
                    "Claim settled within 5 working days",
                ],
                "required_documents": ["RC", "DL", "Policy Copy", "FIR (if applicable)"],
            }
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------------------------------------------
    # Main entrypoint
    # ----------------------------------------------------------------
    def handle_goal(self, user_goal: str, context: dict = None):
        user_goal_lower = user_goal.lower()
        customer_id = context.get("customer_id") if context else None

        try:
            if any(k in user_goal_lower for k in ["verify", "identity", "who am i", "my name"]):
                return self._handle_verification(user_goal, customer_id)

            if any(k in user_goal_lower for k in ["emi", "payment", "due date", "installment", "balance", "paid"]):
                return self._handle_payment(user_goal, customer_id)

            if any(k in user_goal_lower for k in ["claim", "accident", "damage", "theft", "status"]):
                return self._handle_claim(user_goal, customer_id)

            if any(k in user_goal_lower for k in ["sop", "procedure", "insurance", "coverage", "policy"]):
                return self._handle_sop(user_goal, customer_id)

            return "GROC Agent: I couldn't identify the domain for this request. Please clarify: payment, claim, or coverage?"

        except Exception as e:
            error_msg = f"⚠️ GROC Agent Error — {type(e).__name__}: {e}"
            print(error_msg)
            return error_msg

    # ----------------------------------------------------------------
    # Handlers
    # ----------------------------------------------------------------
    def _handle_verification(self, user_goal, customer_id=None):
        if not customer_id or self.customers.empty:
            return "Verification Agent: Please provide your loan number or phone number for verification."

        match = self.customers[
            (self.customers["CustomerID"] == customer_id)
            | (self.customers["LoanID"] == customer_id)
        ]
        if match.empty:
            return "Verification Agent: No matching customer found."
        r = match.iloc[0]
        return f"Verification Agent: Verified {r.get('CustomerName', 'Customer')} (Loan {r.get('LoanID')})."

    def _handle_payment(self, user_goal, customer_id=None):
        """Handle EMI / Payment-related queries."""
        if self.payments.empty:
            return "Payment Agent: No payment data available."
        if not customer_id:
            return "Payment Agent: Please verify your loan ID first."

        # Map verified CustomerID → LoanID
        loan_id = None
        if not self.customers.empty:
            match = self.customers[
                (self.customers["CustomerID"].astype(str).str.lower() == str(customer_id).lower())
                | (self.customers["LoanID"].astype(str).str.lower() == str(customer_id).lower())
            ]
            if not match.empty:
                loan_id = match.iloc[0].get("LoanID")

        # Find in payments using either LoanID or CustomerID
        p = self.payments[
            (self.payments.get("LoanID", pd.Series()).astype(str).str.lower() == str(loan_id).lower())
            | (self.payments.get("CustomerID", pd.Series()).astype(str).str.lower() == str(customer_id).lower())
        ]

        if p.empty:
            return f"Payment Agent: No EMI data found for Loan {loan_id or customer_id}."

        row = p.iloc[0]
        emi_amt = row.get("EMI_Amount", "N/A")
        due_date = row.get("Due_Date", "N/A")
        balance = row.get("Balance", "N/A")
        pay_status = row.get("payment_status", "N/A")
        remarks = row.get("remarks", "")

        return (
            f"Payment Agent: For Loan {loan_id or customer_id}, next EMI is ₹{emi_amt} due on {due_date}. "
            f"Outstanding balance: ₹{balance}. Payment status: {pay_status}. {remarks}"
        )

    def _handle_claim(self, user_goal, customer_id=None):
        """Handle insurance claim status queries."""
        if self.claims.empty:
            return "Claim Agent: No claim data available."
        if not customer_id:
            return "Claim Agent: Please verify your loan ID first."

        # Find loan mapping for claim lookup
        loan_id = None
        if not self.customers.empty:
            match = self.customers[
                (self.customers["CustomerID"].astype(str).str.lower() == str(customer_id).lower())
                | (self.customers["LoanID"].astype(str).str.lower() == str(customer_id).lower())
            ]
            if not match.empty:
                loan_id = match.iloc[0].get("LoanID")

        # Match claim by customer_id (primary key)
        c = self.claims[self.claims["CustomerID"].astype(str).str.lower() == str(customer_id).lower()]
        if c.empty and loan_id:
            c = self.claims[self.claims["CustomerID"].astype(str).str.lower() == str(loan_id).lower()]

        if c.empty:
            return f"Claim Agent: No claim found for Loan {loan_id or customer_id}."

        row = c.sort_values(by="ClaimID").iloc[-1]
        claim_id = row.get("ClaimID", "N/A")
        status = row.get("Status", "N/A")
        amount = row.get("claim_amount", "N/A")
        settlement = row.get("Settlement_Date", "TBD")
        remarks = row.get("remarks", "")

        return (
            f"Claim Agent: Claim ID {claim_id} for Loan {loan_id or customer_id} "
            f"is '{status}'. Claim amount ₹{amount}. Settlement date: {settlement}. {remarks}"
        )

    def _handle_sop(self, user_goal, customer_id=None):
        """Handle SOP and insurance coverage queries."""
        overview = self.sop.get("coverage_overview", "")
        standard = self.sop.get("standard_features", [])
        addons = self.sop.get("common_addons", [])
        steps = self.sop.get("claim_process_steps", [])
        docs = self.sop.get("required_documents", [])

        selected_addons = []
        if customer_id and addons:
            h = int(hashlib.sha256(str(customer_id).encode()).hexdigest(), 16)
            for i in range(min(3, len(addons))):
                idx = (h >> (i * 8)) % len(addons)
                if addons[idx] not in selected_addons:
                    selected_addons.append(addons[idx])

        name = ""
        if not self.customers.empty and customer_id:
            match = self.customers[
                (self.customers["CustomerID"] == customer_id)
                | (self.customers["LoanID"] == customer_id)
            ]
            if not match.empty:
                r = match.iloc[0]
                name = f"For {r.get('CustomerName', 'Customer')} (Loan {r.get('LoanID')}):\n"

        lines = [name + overview, "\nKey standard features:"]
        lines += [f"- {f}" for f in standard]

        if selected_addons:
            lines.append("\nAdd-on covers:")
            for a in selected_addons:
                if isinstance(a, dict):
                    lines.append(f"- {a.get('name')}: {a.get('description')}")
                else:
                    lines.append(f"- {a}")

        lines.append("\nClaim process:")
        lines += [f"- {s}" for s in steps]
        lines.append("\nRequired documents:")
        lines += [f"- {d}" for d in docs]

        return "SOP Agent: " + "\n".join(lines)
