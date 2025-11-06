import pandas as pd

def payment_lookup_tool(customer_id: str) -> str:
    """
    Looks up payment details for a given customer_id.
    """
    df = pd.read_csv("data/payments.csv")

    record = df[df["customer_id"] == customer_id]
    if record.empty:
        return "No payment record found for this customer."

    row = record.iloc[0]
    status = row["payment_status"]
    message = (
        f"💰 Loan Number: {row['loan_number']}\n"
        f"Last Payment: {row['last_payment_date']}\n"
        f"Next Due: {row['next_due_date']}\n"
        f"EMI Amount: ₹{row['emi_amount']}\n"
        f"Outstanding: ₹{row['outstanding_balance']}\n"
    )

    if status.lower() == "overdue":
        message += f"⚠️ Status: Overdue by {row['overdue_days']} days.\nRemarks: {row['remarks']}"
    else:
        message += f"✅ Status: {status}\nRemarks: {row['remarks']}"
    return message
