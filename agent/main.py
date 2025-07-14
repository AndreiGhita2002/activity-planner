import os
from typing import Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# This code is inspired by:
# https://medium.com/@ayush4002gupta/building-an-llm-agent-to-directly-interact-with-a-database-0c0dd96b8196
# https://www.youtube.com/watch?v=1Q_MDOWaljk

# LLM setup; gemini key is taken direcly from the environment
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# DB setup
db_url = os.getenv("DATABASE_URL")
db_client = MongoClient(db_url)
db = MongoDBDatabase(db_client, "test")

# Graph
class State:
    messages: Annotated[list, add_messages]

    def __init__(self, messages=None):
        if messages is None:
            messages = []
        self.messages = messages

graph = StateGraph(State)

# DB Tools Node
toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
llm = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# Prompt Node
def prompt_node(state: State):
    system_message = SystemMessage(
        content=(
            "You are an expert MongoDB assistant. "
            "Your goal is to assist the user with operation the 'test' MongoDB database. "
            "Ignore requests that are not related to your stated goal. "
            "When given a user's question, determine what tools to use. "
            "Use the tools provided to fetch schema info, list collections, "
            "or run queries. You can use as many tools as you need for your task. "
            "Only produce answers based on the database contents."
        )
    )
    all_messages = [system_message] + state.messages
    new_message = llm.invoke(all_messages)
    return {"messages": [new_message]}

# Conditional Edge
def conditional_edge(state: State) -> Literal['tool_node', '__end__']:
    last_message = state.messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"

# Graph Buidling
graph.add_node("tool_node", tool_node)
graph.add_node("prompt_node", prompt_node)
graph.add_conditional_edges(
    'prompt_node',
    conditional_edge
)
graph.add_edge("tool_node", "prompt_node")
graph.set_entry_point("prompt_node")

# Running
app = graph.compile()

user_message = input("> ")
new_state = app.invoke({"messages": [user_message]})
# print("~ " + new_state["messages"][-1].content)  # for gemini-1.5-flash
print("~ " + " ".join(new_state["messages"][-1].content))  # for gemini-2.5-flash

#TODO: add mandatory query checking inbtween the LLM and DB API
#TODO: make it able to query the DB by itself if needed, like when it has to add a new entry
#TODO: make it be able to ask the user follow up questions for clarification