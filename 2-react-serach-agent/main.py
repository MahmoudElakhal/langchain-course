from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()


@tool
def search(query : str) -> str:
    """
    Tool that dearches over the internet
    Args: 
        query : The query to search for
    
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return tavily_client.search(query=query)



llm = ChatOpenAI(model ="gpt-5")
tools = [search]
agent = create_agent(llm, tools)  


def main():
    print("Hello from 2-react-serach-agent!")

    result = agent.invoke({"messages" :HumanMessage(content="Search for 3 job postings for an AI Enginner using langchain in egypt on linkedin and list their details")})
    print(result)
if __name__ == "__main__":
    main()
