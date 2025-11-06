import pandas as pd
from datetime import datetime

def fnol_claim_tool(customer_id: str, incident_type: str = "Accident", remarks: str = "Initial FNOL logged") -> str:
    """
    Simulates logging or checking claim status.
    If claim exists → show latest status.
    If not → create a new one.
    """
    claims_path = "data/claims.csv"
    df = pd.read_csv(claims_path)

    existing = df[df["customer_id"] == customer_id]

    if not existing.empty and (existing.iloc[-1]["claim_status"] in ["New", "In Progress"]):
        row = existing.iloc[-1]
        return (
            f"🧾 Existing Claim Found:\n"
            f"Claim ID: {row['claim_id']}\n"
            f"Type: {row['incident_type']}\n"
            f"Status: {row['claim_status']}\n"
            f"Remarks: {row['remarks']}"
        )

    # Otherwise create a new claim
    new_claim_id = f"CLM{len(df)+1:03d}"
    new_row = {
        "claim_id": new_claim_id,
        "customer_id": customer_id,
        "incident_type": incident_type,
        "incident_date": datetime.now().strftime("%Y-%m-%d"),
        "claim_status": "New",
        "estimated_damage": "",
        "service_center": "",
        "claim_amount": 0,
        "settlement_date": "",
        "remarks": remarks,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(claims_path, index=False)

    return f"✅ New claim {new_claim_id} created for {incident_type}. You can track it later for updates."
