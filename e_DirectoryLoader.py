"""
DirectoryLoader is a document loader that lets you load multiple documents from a directory of files.


"""

from langchain_community.document_loaders import DirectoryLoader,TextLoader

loader = DirectoryLoader(
    # we have to pass three arguments in this load one is path which is the name of directory,other is globe which means which type of files we have to load, and then we have to provide loader_cls which is basically a loader that with the help of which we will load our file like if files are pdfs then we will pass PyPDFLoader etc.
    path="books",
    glob="*.txt" , #it means that load all the txt files only from there
    loader_cls=TextLoader
)

data = loader.load()

# it will gives us an array of document objects and each object will contain two methods metadata and page_content
print(data)