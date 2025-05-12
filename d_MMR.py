"""
Maximal Marginal Relevance (MMR)
10 April 2025 16:24

"How can we pick results that are not only relevant to the query but also different from each other?"

MMR is an information retrieval algorithm designed to reduce redundancy in the retrieved results while maintaining high relevance to the query.

🧐 Why MMR Retriever?

In regular similarity search, you may get documents that are:

    i-All very similar to each other

    ii-Repeating the same info

    iii-Lacking diverse perspectives

MMR Retriever avoids that by:

    i-Picking the most relevant document first

    ii-Then picking the next most relevant and least similar to already selected docs

And so on...

This helps especially in RAG pipelines where:

    I-You want your context window to contain diverse but still relevant information

    II-Especially useful when documents are semantically overlapping

"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


llm = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

from langchain.schema import Document

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

vectorstores = Chroma.from_documents(documents, llm)


retriver = vectorstores.as_retriever(
    search_type="mmr",  #here we will pass search type as mmr for applying mmr strategy for retrieval purpose
    search_kwargs={"k": 3 , "lambda-mult":1} 
    """
    lambda-mult is a balancing factor between:

        i-Relevance to the user's query

        ii-Diversity among the selected documents
    
    What does lambda-mult do?
    MMR tries to select results that are not only relevant but also different from each other.

    If lambda-mult is closer to 1 → Focuses more on relevance (documents will be very related to the query, even if they are a bit repetitive).

    If lambda-mult is closer to 0 → Focuses more on diversity (documents may be less directly relevant but cover different perspectives).
    """
)

query = "Tell me about generative ai"

retriver.invoke(query)