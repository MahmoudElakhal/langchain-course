from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch


load_dotenv()


llm = ChatOpenAI(model ="gpt-5")
tools = [TavilySearch()]
agent = create_agent(llm, tools)  


def main():
    print("Hello from 2-react-serach-agent!")

    result = agent.invoke({"messages" :HumanMessage(content="Search for 3 job postings for an AI Enginner using langchain in egypt on linkedin and list their details")})
    print(result)
if __name__ == "__main__":
    main()
