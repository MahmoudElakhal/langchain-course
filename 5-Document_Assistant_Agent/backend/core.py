
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# core.py is in backend/; ingestion.py is one directory above it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "langchain-docs"  # Must match ingestion.py

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIRECTORY),
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

model = init_chat_model("gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant LangChain documentation from the local Chroma index."""
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return (
            "No relevant documentation found in the local Chroma database. "
            "Run ingestion.py to populate the database first.",
            [],
        )

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


system_prompt = (
    "You are a helpful AI assistant that answers questions about LangChain documentation. "
    "Use the retrieve_context tool to find relevant documentation before answering. "
    "Always cite the URLs you use as sources. "
    "If the indexed documentation does not contain the answer, say so."
)

# Construct this once rather than once per user question.
agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)


def run_llm(query: str) -> Dict[str, Any]:
    """Answer a question and return documents used for Streamlit's Sources section."""
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and isinstance(message.artifact, list):
            context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    print(run_llm(query="What are deep agents?"))
