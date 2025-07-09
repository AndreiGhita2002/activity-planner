import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# This file takes inspiration from:
# https://medium.com/@ayush4002gupta/building-an-llm-agent-to-directly-interact-with-a-database-0c0dd96b8196

# LLM setup; gemini key is taken direcly from the environment
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# DB setup
db_url = os.getenv("DATABASE_URL")
db_client = MongoClient(db_url)
db = MongoDBDatabase(db_client, "test")

# DB tools
toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
info_tool = next((tool for tool in tools if tool.name == "mongodb_schema"), None)
list_collections_tool = next((tool for tool in tools if tool.name == "mongodb_list_collections"), None)
query_tool = next((tool for tool in tools if tool.name == "mongodb_query"), None)
query_checker_tool = next((tool for tool in tools if tool.name == "mongodb_query_checker"), None)


def make_query(request: str):
    #
    pass


# testing
print("info_tool: \n", info_tool.invoke("test"))
print("list_collections_tool: \n", list_collections_tool.invoke(""))
print("query_tool(db.test.find({value:27})): \n", query_tool.invoke("db.test.find({value:27})"))
print("query_checker_tool(db.test.find({value:27})): \n", query_checker_tool.invoke("db.test.find({value:27})"))
