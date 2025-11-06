# ===========================================
# agent_logic_llm.py
# ===========================================

from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Import existing tools
from tools.verify_user_tool import verify_user_tool
from tools.payment_lookup_tool import payment_lookup_tool
from tools.fnol_claim_tool import fnol_claim_tool
from tools.sop_lookup_tool import sop_lookup_tool
from tools.crm_logger_tool import crm_logger_tool
from llm_loader import load_llm

class AutoFinanceLLMAgent:
    """
    A single-step reasoning agent powered by a local Ollama LLM.
    The LLM decides which tools to call based on the user's query.
    """

    def __init__(self, model_name="mistral"):
        # Connect to local Ollama LLM
        # self.llm = Ollama(model=model_name)
        self.llm = load_llm()

        # Register tools
        self.tools = [
            Tool(
                name="Verify User",
                func=verify_user_tool,
                description="Verify customer identity using name, loan number, or contact details."
            ),
            Tool(
                name="Payment Lookup",
                func=payment_lookup_tool,
                description="Fetch EMI, payment due date, and loan balance for verified customers."
            ),
            Tool(
                name="FNOL Claim Tool",
                func=fnol_claim_tool,
                description="Handle accident, theft, or damage insurance claims."
            ),
            Tool(
                name="SOP Lookup",
                func=sop_lookup_tool,
                description="Retrieve process or policy information from SOP documents."
            ),
            Tool(
                name="CRM Logger",
                func=crm_logger_tool,
                description="Log chat messages to CRM system for future reference."
            ),
        ]

        # Initialize LLM-powered agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def route_query(self, user_input: str) -> str:
        """
        Pass the user's query to the LLM-powered agent.
        """
        try:
            response = self.agent.run(user_input)
        except Exception as e:
            response = f"⚠️ LLM Agent Error: {str(e)}"

        # Log all queries and responses
        crm_logger_tool(user_input, response)
        return response
