"""
Contextual Compression Retriever:-
The Contextual Compression Retriever in LangChain is an advanced retriever that improves retrieval quality by compressing documents after retrieval — keeping only the relevant content based on the user's query.

❓ Query:
    “What is photosynthesis?”

📄 Retrieved Document (by a traditional retriever):

    “The Grand Canyon is a famous natural site.
    Photosynthesis is how plants convert light into energy.
    Many tourists visit every year.”

✖️ Problem:

The retriever returns the entire paragraph

    i-Only one sentence is actually relevant to the query

    ii-The rest is irrelevant noise that wastes context window and may confuse the LLM



✅ What Contextual Compression Retriever does:

Returns only the relevant part, e.g.
    "Photosynthesis is how plants convert light into energy."

⚙️ How It Works

    i-Base Retriever (e.g., FAISS, Chroma) retrieves N documents.
    ii-A compressor (usually an LLM) is applied to each document.
    iii-The compressor keeps only the parts relevant to the query.
    iv-Irrelevant content is discarded.
    
✅ When to Use

    a)Your documents are long and contain mixed information
    b)You want to reduce context length for LLMs
    c)You need to improve answer accuracy in RAG pipelines


"""

# !pip install langchain chromadb langchain-huggingface langchain-community langchain-google-genai google-generativeai

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
gemini_token = os.getenv("GEMINI_API_KEY")
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_token
)
embeding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token = hf_token
)


# create a vector store
# here we are creating an instance of a vector store using the Chroma vector store implementation.We have seen in previoud cahpter of vector store that how to create a store.
vector_store = Chroma(
    embedding_function=embeding_model,
    persist_directory="vectors",
    collection_name="temporary"
)


documents = [
    Document(
        page_content="Albert Einstein developed the theory of relativity. Pizza is a popular Italian dish. Quantum mechanics describes the behavior of particles at microscopic scales.",
        metadata={"source": "doc1"}
    ),
    
    Document(
        page_content="Mount Everest is the tallest mountain on Earth. Photosynthesis enables plants to make food using sunlight. Basketball is played with five players on each team.",
        metadata={"source": "doc2"}
    ),
    Document(
        page_content="The French Revolution began in 1789. Hydrogen is the lightest element in the periodic table. JavaScript is commonly used for web development.",
        metadata={"source": "doc3"}
    ),
    Document(
        page_content="Shakespeare wrote many famous plays. The mitochondrion is the powerhouse of the cell. Ice cream melts quickly on a hot day.",
        metadata={"source": "doc4"}
    )
]

# add some data in store
vector_store.add_documents(documents)

# create a retriver for getting data from vector store.We can create a retriever from our vector store that helps us in fetching data relevent to given query.For this purpose we use as_retriever method of vector store in which we pass an argument search_kwargs which is basicaly a dict that contains how many relevant answers we want to get from retriever.
retriever = vector_store.as_retriever(search_kwargs={'k':2})


# now we have to create a compressor that will compress the output of retriever.It will receive output of retriever and will see if any content in given output is irrelevant to given query then it remove that part only.We create it as follows , and we have to pass a model to it as an argument which basically do the process of compression.
compresser = LLMChainExtractor.from_llm(gemini_model)

# then at last we call our ContextualCompressionRetriever which take two args one is base_compressor and other is base_retriever,we assign compresser to base_compressor and retriever to base_retriever.It is like a runnable that combines both retriever and compressor together and takes work from them.
compressor_retriver = ContextualCompressionRetriever(base_compressor=compresser , base_retriever=retriever)



# now we can send our query direct to this compressor retriver
query = "tell me about pizza"
result = compressor_retriver.invoke(query)
# this result will be containing an array of langchain documents in which our main result will be present.
print(result) 

# There are multiple retrievers present in langchain , we can use any of them on our need base.We can exolore retriever on langchain docs.