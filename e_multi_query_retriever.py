"""
Multi-Query Retriever:-
    Sometimes a single query might not capture all the ways information is phrased in your documents.

For example:
Query:
"How can I stay healthy?"

Could mean:

    i-What should I eat?

    ii-How often should I exercise?

    iii-How can I manage stress?

A simple similarity search might miss documents that talk about those things but don’t use the word “healthy.”

So how Multi-Query Retriever helps us in this regard
    i-Takes your original query

    ii-Uses an LLM (e.g., GPT-3.5) to generate multiple semantically different versions of that query

    iii-Performs retrieval for each sub-query

    iv-Combines and deduplicates the results

Like if user gives query :  
"How can I stay healthy?"
 then multi-query retriver will generate such type of multiple queries:
"What are the best foods to maintain good health?"

"How often should I exercise to stay fit?"

"What lifestyle habits improve mental and physical wellness?"

"How can I boost my immune system naturally?"

"What daily routines support long-term health?"


"""

from langchain.retrievers import MultiQueryRetriever
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    google_api_key = gemini_key
)

embeding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)
documents = [
    Document(
        page_content="""
LangChain is a framework for building LLM-powered applications using composable components like prompts, memory, chains, tools, and retrievers. 
It is especially popular for Retrieval-Augmented Generation (RAG) use cases and agent development.
""",
        metadata={"source": "LangChain Docs", "topic": "LangChain"}
    ),
    Document(
        page_content="""
LangGraph is a graph-based extension of LangChain that supports stateful, multi-step reasoning. 
It allows for flexible control flow, making it ideal for building complex agent workflows and decision trees.
""",
        metadata={"source": "LangGraph Docs", "topic": "LangGraph"}
    ),
    Document(
        page_content="""
CrewAI is a framework that enables the creation of multi-agent systems where agents have roles, goals, and tools. 
Agents can collaborate to solve complex tasks, making CrewAI well-suited for autonomous agent orchestration.
""",
        metadata={"source": "CrewAI Docs", "topic": "CrewAI"}
    ),
    Document(
        page_content="""
Generative AI refers to models that can produce text, images, code, or other content. 
LLMs like GPT are core to this space and can be integrated with tools like LangChain for building intelligent applications.
""",
        metadata={"source": "GenAI Overview", "topic": "Generative AI"}
    )
    ,
    Document(
        page_content="""
Generative AI refers to models that can produce text, images, code, or other content. 
LLMs like GPT are core to this space and can be integrated with tools like LangChain for building intelligent applications.
""",
        metadata={"source": "GenAI Overview", "topic": "Generative AI"}
    )
]

vectorstores = Chroma.from_documents(documents, embeding_model)

multi_query_retriver = MultiQueryRetriever.from_llm(
    retriever=vectorstores.as_retriever(),
    llm=llm,
)

query = "tell me about a type of AI"
response = multi_query_retriver.invoke(query)
print(response)