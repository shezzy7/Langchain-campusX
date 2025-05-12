"""
A wikipedia retriver is a retriver that queries the Wikipedia API to fetch the relevant content for a given query.
How it Works ->
    i-You give it a query
    ii-It sends the query to wikipedia's API
    iii-It retrieves the most relevant articles.
    iv-It returns them as langchain Document objects.
"""

from langchain_community.retrievers import WikipediaRetriever

# we have to pass two args in wikipediaretriever.One is how many relevant results we want to get,and second one is name of language in which we want results
retriever = WikipediaRetriever(
    top_k_results=2,
    lang='en'
    
)

query = "the geological history of india and pakistan from the perspective of a chinese"

docs = retriever.invoke(query) # it will return a list of langchain documents

for idx , doc in enumerate(docs):
    print("index - >"+str(idx)+"\n")
    print("Content ->"+ doc.page_content)


