# Length Based splitter splits the text on the basis of given number of characters.If ask him to split the text and each chunk shoul contain 100 characters.

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text = "Artificial Intelligence (AI) is transforming industries by enabling machines to perform tasks that typically require human intelligence. From autonomous vehicles to personalized healthcare, AI applications are becoming increasingly prevalent. Natural Language Processing (NLP), a subset of AI, allows computers to understand and generate human language. Technologies like ChatGPT and BERT are examples of large language models that demonstrate the power of NLP. As these tools continue to evolve, ethical concerns around data privacy, bias, and transparency are also becoming more significant. Understanding how to responsibly build and deploy AI systems is essential for sustainable progress."

splitter = CharacterTextSplitter(
    # it takes three params as input
    # one is the length of each chunk we want him to generate
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

chunks = splitter.split_text(text) #it will return us a list of strings,each string will contain 100 characters except last chunk that may contain less than 100 characters.
print(chunks)

# lets load a text file and apply splitting on it



loader = TextLoader('a_textSplitter.txt')

loaded_text = loader.load()

chunks_of_loaded_text = splitter.split_text(loaded_text[0].page_content)

for chunk in chunks_of_loaded_text:
    print(chunk)
# or we can use split_documents bcz our text is present in an array of Document object.

chunks_of_loaded_text2 = splitter.split_documents(loaded_text)
