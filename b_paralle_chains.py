# Here we are going to build an application in which we will be using a parallel chain.Parallel chain mean multiple chain will be working togethere.
# Here we will get notes on a topic from a llm and will use another model for generating quize(mcqs) on this topic and both these models will be working paralley.And the we will be merging both notes and quiz together using one of the above models.


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel #we will use this component for creating a parallel chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
hf_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
model1 = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash-lite",
    google_api_key=gemini_api_key
)

# hug_llm = HuggingFaceEndpoint(
#     repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task = 'text-generation',
#     huggingfacehub_api_token=hf_api_key
# )
# model2 = ChatHuggingFace(llm=hug_llm)

model2 = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
    google_api_key = gemini_api_key
)
parser = StrOutputParser()

template1 = PromptTemplate(
    template="""
        Generate consice and easy to understand notes on {topic}
    """,
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="""
        Generate some mcq type quiz on {topic}
    """,
    input_variables=['topic']
    
)

template3 = PromptTemplate(
    template="""
        Merge these notes and quiz into a single document \n notes ->{notes} , quiz ->{quiz}
    """ ,
    input_variables=['notes','quiz']
)
# For making a parallel chain we use RunableParallel and we pass chains in the form of dictionary to it.Key is name of chain and value is original

parallel_chain = RunnableParallel(
    {
    
        'notes' : template1 | model1 | parser,
        'quiz' : template2 | model2 | parser
    }
) 

# lets build another chain that will merge both of these results
merge_chain = template3 | model1 | parser

# we will make another chain for combing both of above chain together

chain = parallel_chain | merge_chain

topic_name = input("Enter name of topic : " )
response = chain.invoke({'topic':topic_name})
print(response)