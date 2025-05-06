# JsonOutputParser in LangChain is a tool that helps us reliably extract structured information from the output of a language model in JSON format.We tell the language model to give us its answer as JSON, and the JsonOutputParser makes sure us get a valid Python dictionary (or list of dictionaries) from that JSON string.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    google_api_key=gemini_api_key
)

parser = JsonOutputParser()

template =  PromptTemplate(
    template = 'Write name,age,contact and location(including country name) of a fictional person.Also tell what is a fictional person \n {formate_instruction}',
    input_variables=[],
    # here we are using partial_variables becuase we are not using any variable that will be inputed from user during runtime.Here we just passing formate_instruction whose value will be determined by our parser and we call such type of variable partial_variables.A method of parser known as get_formate_instruction generates the instructions for llm for generating a response in json formate.
    partial_variables={'formate_instruction':parser.get_format_instructions()}
)


# prompt = template.format() #here we are calling format method of template instead of invoke(as we do earlier) bcz invoke is called with some input but here our template does not need any input from user so we are calling its format method whose functionality is as same as of invoke.

# response = model.invoke(prompt)
# final_output = parser.parse(response.content)#after getting response from model we will parse it.
# print(final_output)

# we can use chain for concising our code as below
chain = template | model | parser 
result = chain.invoke({})   #our invoke method always gets an argument for working , and here we don't have any input to give , so we can pass an empty dictionary and it will work properly else it will through an error.
print(result)

# there is a flaw in jsonOutputParser which is that we can't define our own schema for model's output.Model defines by itself that what keys it will have.So this problem can be solved by StructuredOutputParser.