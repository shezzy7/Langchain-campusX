# RunnableParallel -> It is a runnable primitive that allows multiple runnables to execute in parallel.Each runnable receives the same input and processes it independently,producing a dictionary of outputs.
# for example if we want to generate a joke and a linked post on the same topic then we can use two different models combined through RunnableParallel.And to this runnable we will be giving same input but both of these will run at same time one will generate joke and other will generate a linkedin post.

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel , RunnableSequence

load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")

hf_token = os.getenv("HUGGINGFACE_API_KEY")
model1 = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
    google_api_key = gemini_api_key
)

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    huggingfacehub_api_token=hf_token
)

model2 = ChatHuggingFace(llm=llm)
template1 = PromptTemplate(
    template="""
        Generate joke for given topic : {topic}
    """,
    input_variables=['topic']
)

parser = StrOutputParser()

template2 = PromptTemplate(
    template = """
        Write a tweet about -> {topic}
    """,
    input_variables=['topic']
)



chain = RunnableParallel({
    'joke' : RunnableSequence(template1 , model1 , parser) ,
    'tweet' : RunnableSequence(template2 , model1 , parser)
    # here my model2 was running slow ,due to which an error was being generated so i have used same model in both chains.
})

response = chain.invoke({'topic' : "Dubai"})

print(response)   