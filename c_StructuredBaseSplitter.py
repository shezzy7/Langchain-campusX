# We can split our text on the basis of its structure.
# In this splitter we have to pass chunk size,and overlap_size.
# It will work in such a way that it will see wheather the whole text's length is greater then given chunk size if so then it will find is there any double line space in given text like '\n\n' , on finding it will split the text before and after it into two  or according to number of paragraphs separated by two line spaces.Then after spliting.It will check wheather there are any chunk whose length is greater then given chunk size if so it will split that chunk on the basis of single line space like '\n' and later on it go on splitting by word then comma and so on

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

text = """Artificial Intelligence (AI) is transforming industries by enabling machines to perform tasks that typically require human intelligence. From autonomous vehicles to personalized healthcare, AI applications are becoming increasingly prevalent.
Natural Language Processing (NLP), a subset of AI, allows computers to understand and generate human language. Technologies like ChatGPT and BERT are examples of large language models that demonstrate the power of NLP. As these tools continue to evolve, ethical concerns around data privacy, bias, and transparency are also becoming more significant. Understanding how to responsibly build and deploy AI systems is essential for sustainable progress."""
data = splitter.split_text(text)

print(data)