# ===========================================
# tools/verify_user_tool.py (CSV Header-Aware Version)
# ===========================================

import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
CUSTOMER_FILE = os.path.join(DATA_PATH, "customers.csv")


def _load_customers():
    """Load customer master data flexibly across different header styles."""
    if not os.path.exists(CUSTOMER_FILE):
        return pd.DataFrame()

    df = pd.read_csv(CUSTOMER_FILE, dtype=str).fillna("")

    # Normalize all possible header variants
    rename_map = {}
    if "customer_id" in df.columns and "CustomerID" not in df.columns:
        rename_map["customer_id"] = "CustomerID"
    if "first_name" in df.columns and "last_name" in df.columns and "CustomerName" not in df.columns:
        # Merge first + last name to CustomerName
        df["CustomerName"] = df["first_name"].astype(str) + " " + df["last_name"].astype(str)
    if "loan_number" in df.columns and "LoanID" not in df.columns:
        rename_map["loan_number"] = "LoanID"
    if "phone" in df.columns and "Phone" not in df.columns:
        rename_map["phone"] = "Phone"
    if "vehicle_model" in df.columns and "Vehicle" not in df.columns:
        rename_map["vehicle_model"] = "Vehicle"
    if "registered_city" in df.columns and "City" not in df.columns:
        rename_map["registered_city"] = "City"

    # Apply rename
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def verify_user_tool(query: dict):
    """
    query: dict with any of {'name': 'John Doe', 'loan': 'LN001', 'phone': '9999999990'}
    Returns:
        {'ok': True, 'customer_id': 'C001', 'first_name': 'John', 'last_name': 'Doe', ...}
        or {'ok': False, 'reason': 'not found'}
    """
    df = _load_customers()
    if df.empty:
        return {"ok": False, "reason": "no customer data available"}

    name = str(query.get("name", "")).strip().lower()
    loan = str(query.get("loan", "")).strip().lower()
    phone = str(query.get("phone", "")).strip()
    matched = pd.DataFrame()

    # --- Search by Loan ID ---
    if loan:
        if "LoanID" not in df.columns:
            return {"ok": False, "reason": "loan id column missing in data"}
        matched = df[df["LoanID"].astype(str).str.lower() == loan]

    # --- Search by Phone ---
    elif phone:
        def norm_phone(p):
            return "".join([c for c in str(p) if c.isdigit()])

        if "Phone" in df.columns:
            phone_series = df["Phone"].astype(str)
        else:
            phone_series = pd.Series([""] * len(df))

        df["__norm_phone"] = phone_series.apply(norm_phone)
        norm_query_phone = norm_phone(phone)
        matched = df[df["__norm_phone"] == norm_query_phone]
        df.drop(columns=["__norm_phone"], inplace=True, errors="ignore")

    # --- Search by Name ---
    elif name:
        if "CustomerName" not in df.columns:
            return {"ok": False, "reason": "name column missing in data"}
        matched = df[df["CustomerName"].astype(str).str.lower().str.contains(name)]

    else:
        return {"ok": False, "reason": "no verification info provided"}

    if matched.empty:
        return {"ok": False, "reason": "not found"}

    row = matched.iloc[0].to_dict()
    full_name = row.get("CustomerName", "")
    parts = full_name.split(" ", 1)
    first = parts[0] if parts else ""
    last = parts[1] if len(parts) > 1 else ""

    return {
        "ok": True,
        "customer_id": row.get("CustomerID") or row.get("LoanID"),
        "first_name": first,
        "last_name": last,
        **row,
    }
