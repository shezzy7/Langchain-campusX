"""
TextLoader is a simple and commonly used document loader in a langchain that reads plain text(.txt) files and converts them into langchian document objects.

"""

# All the document loaders are present in community package of langchain

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    google_api_key = gemini_key
)

# for loading data from a txt file we pass the path of the file in TextLoader and save it in some variable
loader = TextLoader("poem.txt")

# insider loader there is method named as load we have to call it for getting main document.It will return us an array.Each element of array will be an object and inside this object two methods are present.One is metadata which basically contains the name of the file and other is page_content which contains main data of file.As here we are uploading only single file so array will contain only single element which we access by using index 0.
data = loader.load()

print(data[0].page_content)

# we can perform any kind of work on this data now.Like we can generate summary of this data using a model


template = PromptTemplate(
    template="""
        Generate a summary of the following poem -> {poem}
    """,
    input_variables=['poem']
)

parser = StrOutputParser()
chain = template | model | parser

print(chain.invoke({'poem' : data[0].page_content}))