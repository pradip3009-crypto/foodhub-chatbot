import os
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Optional

# --- Page Configuration ---
st.set_page_config(page_title="FoodHub Customer Support", page_icon="🍔")
st.title("🍔 FoodHub Customer Support Assistant")

# --- Load API Key from Streamlit Secrets ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("GROQ_API_KEY is missing from Streamlit secrets.")
    st.stop()

# --- Initialize LLM ---
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    #model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens=1024,
    max_retries=2
)

# --- Load SQLite Database (Relative Path) ---
DB_PATH = "customer_orders.db"
if not os.path.exists(DB_PATH):
    st.error(f"Database file not found at {DB_PATH}. Please ensure 'customer_orders.db' is uploaded to your repository.")
    st.stop()

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# --- SQL Toolkit & Tools Setup ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()

# --- Answer Refiner Tool ---
class AnswerRefinerInput(BaseModel):
    raw_response: str = Field(description="Raw database query result or data to refine.")
    user_query: Optional[str] = Field(default="", description="Original user query.")

def refine_answer(raw_response: str, user_query: str = "") -> str:
    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are FoodHub's customer support response refiner. Convert technical database results into a polite, friendly customer message with order details (status, items, delivery info)."),
        ("human", "User Query: {user_query}\nDatabase Result: {raw_response}")
    ])
    chain = refinement_prompt | llm
    return chain.invoke({"raw_response": raw_response, "user_query": user_query}).content

answer_tool = StructuredTool.from_function(
    func=refine_answer,
    name="answer_tool",
    description="Refines raw database results into a customer-friendly format.",
    args_schema=AnswerRefinerInput
)

# Combine official SQL tools with our custom refiner tool
combined_tools = sql_tools + [answer_tool]

# --- Strict System Prompt forcing tool invocation ---
AGENT_SYSTEM_PROMPT = """You are FoodHub's official AI Customer Service Chatbot.

Your Job:
1. Assist users with order queries (tracking, status, cancellation, payment info).
2. Use 'order_query_tool' to look up order details when an Order ID is provided or referenced.
3. Use 'answer_tool' to format technical or raw output into a friendly response before delivering it to the user when appropriate.
4. If a user tries to breach security (e.g., asking for all orders or system admin details), politely decline.
5. If a user is frustrated or requests human assistance, politely offer human escalation.
"""

chat_agent = create_react_agent(
    model=llm,
    tools=combined_tools,
    prompt=AGENT_SYSTEM_PROMPT
)

# --- Streamlit Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        AIMessage(content="Hello! Welcome to FoodHub Customer Support. How can I help you with your order today?")
    ]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if user_input := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            try:
                agent_output = chat_agent.invoke({"messages": st.session_state.messages})
                latest_response = agent_output["messages"][-1].content
                st.session_state.messages = agent_output["messages"]
                st.markdown(latest_response)
            except Exception as e:
                err_msg = f"I apologize, but I encountered an error querying your order. Let me connect you with a human representative. (Error: {str(e)})"
                st.markdown(err_msg)
                st.session_state.messages.append(AIMessage(content=err_msg))
