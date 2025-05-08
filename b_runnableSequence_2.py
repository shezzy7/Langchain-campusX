# There are multiple Runnable Primitives some of them are below.

# RunnableSequence :- It is a sequential chain of runnables in langchain that executes each step one after another,passing the output of one step as a input for next step.
# It is useful when you need to compose multiple runnables together in a structured workflow.

# let's do an example of RunnableSequence where we will be generating a template for giving input to model and then model's output will be going to parser,then we will pass parser's output to a template2 which will generating a prompt for model for explaining given and then model will be invoked with this input and finally we will send model's output to a parser which will give us final output 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key=os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
    google_api_key = gemini_api_key
)

template1 = PromptTemplate(
    template="""
        Generate joke for given topid : {topic}
    """,
    input_variables=['topic']
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template = """
        Explain the following joke -> {joke}
    """,
    input_variables=['joke']
)

# now we will be combining all the runnables in RunnableSequence

chain = RunnableSequence(template1 , model  , parser , template2 , model , parser )
# we will be passing are the runnables in sequence separated by commas.
topic = input("Enter topic on which you want to generate a joke: ")
response = chain.invoke({'topic' : topic })
print(response)
