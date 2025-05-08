# RunnableLambda -> It is a runnable primitive that allows us to apply custome Python functions within an AI pipeline.
# It acts as a middleware between different AI components,enabling preprocessing,transformation,API calls,filtering and post-processing in a langchain workflow.

# For example we have some data of comments on youtube and we want to make sentiment of those comments.But before sentiment we want to filter out emojis,quotation marks from them before sending them to our model.So for this purpose we can write a function which performs this task of preprocessing and make this function  a runnable with the help of RunnableLambda.So that we can use this function as a runnable.Once it has become a runnable we can use it in any chain(bcz chaining can be applied only on runnables.)

# Here lets do an activity where we want to generate a joke from llm and we will creat a function which will count number of words in this joke and we will be using it in our chains by making it a runnable through RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableLambda , RunnableParallel , RunnablePassthrough
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



def word_count(text):
    return len(text.split())

topic = input("Enter topic on which you want to generate a joke: ")
joke_gen = RunnableSequence(template1 , model , parser)

parallel_chain = RunnableParallel(
    {
        'joke' : RunnablePassthrough() , 
        'words' : RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_gen , parallel_chain)

response = final_chain.invoke({'topic' : topic})

print(response['joke'])
print(response['words'])
