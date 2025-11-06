# 🏦 Auto Finance Agentic Assistant (Groq + LangChain + Streamlit)

### 🤖 A fully local, intelligent **Auto Finance Helpdesk Assistant** powered by **Groq LLM** and local multi-agent orchestration.

This project demonstrates an **Agentic AI System** that automates customer verification, EMI management, insurance coverage explanation, and claim tracking — using local CSV files and LLM-based reasoning.

---

## 🚀 Key Features

✅ **Customer Verification**
- Verifies user identity from `customers.csv` using loan number or phone number.
- Retains session memory after verification (no re-verification needed).

✅ **Payment (EMI) Agent**
- Retrieves EMI amount, due date, outstanding balance, and payment status.
- Supports questions like _“When is my next EMI due?”_ or _“Is my EMI paid?”_

✅ **Claim Agent**
- Retrieves claim ID, status, amount, settlement date, and remarks from `claims.csv`.

✅ **Insurance SOP Agent**
- Explains insurance coverage, standard features, add-ons, and claim process steps.

✅ **Supervisor Agent (LLM-Powered)**
- Plans which sub-agent to trigger, executes actions, and reflects on completion status.

✅ **Audit Logging (CRM Tool)**
- All interactions are logged automatically in `logs/chat_history.log`.

✅ **Data Privacy**
- All data operations are local (CSV + JSON). Only LLM reasoning runs on Groq Cloud.

---

## 🧠 System Architecture

```
┌──────────────────────────┐
│        Streamlit UI      │  ← User enters chat (hi, LN001, claim status)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   🧠 Supervisor Agent     │  ← Plans task (verify → payment/claim/sop)
└────────────┬─────────────┘
             │
     ┌───────┼──────────────────────────────┐
     ▼       ▼              ▼               ▼
🧾 Verify   💰 Payment     🧾 Claim        🧩 SOP
Agent      Agent          Agent           Agent
  │           │              │               │
  ▼           ▼              ▼               ▼
customers.csv payments.csv  claims.csv      sop.json
             │
             ▼
   ✅ GROC (Goal Reasoning Orchestration Controller)
```

---

## 🗂️ Folder Structure

```
auto_finance_agentic/
│
├── app.py                        # Streamlit frontend
├── supervisor_agent.py            # LLM-powered task planner + reflection
├── agent_groc.py                  # Executes goals using CSV data
│
├── tools/
│   ├── verify_user_tool.py        # Customer verification tool
│   ├── crm_logger_tool.py         # Logging and chat tracking
│
├── data/
│   ├── customers.csv              # Customer master data
│   ├── payments.csv               # EMI data
│   ├── claims.csv                 # Insurance claims
│   ├── sop.json                   # SOP and coverage details
│
├── logs/
│   └── chat_history.log           # All user interactions logged here
│
├── llm_loader.py                  # Loads Groq API model securely
├── requirements.txt               # Python dependencies
└── .env                           # Contains your GROQ_API_KEY
```

---

## ⚙️ Installation

### 🪟 Prerequisites
- Python **3.10+**
- Streamlit **1.39.0+**
- Groq API key (free tier works fine)

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/<your-username>/auto-finance-agentic.git
cd auto-finance-agentic
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv agentic-env
agentic-env\Scripts\activate   # (on Windows)
# or
source agentic-env/bin/activate  # (on macOS/Linux)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Groq API Key

Create a `.env` file in the project root:

```bash
GROQ_API_KEY="your_api_key_here"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

App will launch on:
```
👉 http://localhost:8501
```

---

## 💬 How to Test

### Step 1: Start the Assistant
Type:
```
hi
```

### Step 2: Verify Yourself
```
LN001
```

### Step 3: EMI Queries
```
is my EMI paid?
when is my next EMI due?
```

### Step 4: Claims Queries
```
claim status
update my claim
```

### Step 5: Insurance Coverage
```
explain my insurance coverage
what does my policy include?
```

### Step 6: Review Logs
Open:
```
logs/chat_history.log
```
