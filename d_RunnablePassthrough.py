# RunnablePassthrough -> It is a runnable  that return the input as output without modifying it.It is used in special conditions like.In prvious file b_runnableSequence.py we are creating a chain that takes topic name as input and in this chain a model generates a joke on this topic and then this joke is sended to other model for its explanation and then finaly we get the explanation of the joke.But we don't know what was original joke bcz we have only explanation of that joke.In this scenarion RunnablePassthrough helps us to get our joke also
# lets do this

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough
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
        Generate joke for given topic : {topic}
    """,
    input_variables=['topic']
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template = """
        Explain the following joke in concise and simple way   -> {joke}
    """,
    input_variables=['joke']
)

# now we will be combining all the runnables in RunnableSequence

chain1 = RunnableSequence(template1 , model  , parser )
chain2 = RunnableParallel(
    {
        'joke' : RunnablePassthrough() , #this simply return the given input as output which will be stored in joke key
        'explanation' : RunnableSequence(template2 , model , parser)
    }
) 
# we will be passing are the runnables in sequence separated by commas.

final_chain = RunnableSequence(chain1 , chain2)
topic = input("Enter topic on which you want to generate a joke: ")
response = final_chain.invoke({'topic' : topic })
print(response['joke']+"\n")
print(response['explanation']+"\n")


