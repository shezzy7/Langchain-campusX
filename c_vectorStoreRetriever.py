""" A Vector Store Retriever in langchain is the most common type of retriever that lets you search and fetch documents from a vector store based on semantic similarity using vector embeddings

How it works :-
    i-You store your documents in a vector store(like FAISS , Chroma , Weaviate)
    ii-Each document is converted into dense vector using an embedding mode
    iii-When the user enters a query:
        a) It's also turned into a vector
        b) The retriever compares the query vector with the stored vectors
        c)It retrieves teh top-k most similar ones
         

"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

llm = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


docs = [
    Document(page_content="Langchain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="OpenAI provides powerful embedding models")
]

vectorstore = Chroma(
    embedding_function=llm,
    persist_directory="chroma_db",
    collection_name="retriver1"
)

vectorstore.add_documents(docs)

# we can create a retriever using our vectorstore , as in our vectorstore a method named as as_retriever is called in which we pass an argument search_kwargs which is basically a dict in this dict we have to pass value of k , which is number of results we our retriever generate for us

retriever = vectorstore.as_retriever(search_kwargs={'k':2})


response = retriever.invoke('what is use of chroma?')
print(response)