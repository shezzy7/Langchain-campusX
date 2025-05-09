# WebBaseLoader is a document loader in langchain used to load and extract text content from web pages(URLs)
# it uses BeautifulSoup under the hood to parse HTML and extrat visible text.
# It converts all the static data(text) of web into a sinlge document object and return an array that contains that object.

from langchain_community.document_loaders import WebBaseLoader

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




template = PromptTemplate(
    template="""
        Answer the following question -> {question} \n from given text ->{text}
    """,
    input_variables=['question' , 'text']
)

parser = StrOutputParser()

# here we will be passing a website to loader and then asking questin about the data of that site.

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Muhammad")

data = loader.load()

chain = template | model | parser

result = chain.invoke({"question" : 'Tell me his parents name' , 'text':data[0].page_content})
print(result)