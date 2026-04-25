from typing import List
from pydantic import BaseModel  ,Field

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

load_dotenv()

# This Class is going to represent the source of the answer
class Source(BaseModel):

    """ Schema for a source used by the agent
    """    
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Scehma for agent response with answer and sources"""
    answer:str = Field(description = "The Agent's answer to the query") 
    sources:List[Source] = Field(default_factory = list , description = "List of Sources used to generate the answer")



llm = ChatOpenAI(model ="gpt-5")
tools = [TavilySearch()]
agent = create_agent(llm, tools , response_format = AgentResponse)  


def main():
    print("Hello from 2-react-serach-agent!")

    result = agent.invoke({"messages" :HumanMessage(content="Search for 3 job postings for an AI Enginner using langchain in egypt on linkedin and list their details")})
    print(result)
if __name__ == "__main__":
    main()
