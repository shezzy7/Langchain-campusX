"""
Chain:-
A Chain in LangChain is a way to connect multiple steps or components (like prompts, models, tools, etc.) to work together in a sequence.

Instead of calling an LLM directly with just one prompt, a Chain helps you build more structured workflows — where inputs are processed, passed through the model, and outputs are formatted or used in the next step.

Types of Chains:
    i-Simple Chain: One prompt → One response

    ii-Sequential Chain: Multiple steps, where output of one becomes input of the next

    iii-Custom Chain: You create your own logic using LangChain's building blocks
    
Example :-
You can create a chain that:

i-Takes user input

ii-Formats it into a prompt

iii-Sends it to a model (like GPT or Gemini)

iv-Processes the response

v-Returns it in a nice format (like JSON, text, or UI output)
""" 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash-lite',
    google_api_key = gemini_api_key
)

template1 = PromptTemplate(
    template = "Give me a detailed record on the topic {topic} ",
    input_variables=['topic']
)

topic = input("Enter name of topic: ")

template2 = PromptTemplate(
    template = "Give me a summary of {text} \n there is no need of adding any text  which show that you are summarizing a text given to you.",
    input_variables=['text']
) 

parser = StrOutputParser()

# here we want to get detailed record on a topic entered by the user ,and we will be using template1 for generating prompt,this prompt will be sended to model and model will give us a response then we will send this response's content to parser for extracting content from response and then send this parser's output  to template2 for generating next prompt for model and then will call our model again for getting desired output,again we will send this response to parser for extracting content from it.
# So instead of doing this manually in steps we set all of these steps in a chain which will do all of the tasks in a sequence and we call such type of chain as sequential chains.

chain  = template1  | model | parser | template2 | model | parser
response = chain.invoke({'topic' : {topic}})
print(response)

# we can also print workflow of our chain in following way
print(chain.get_graph().print_ascii())
