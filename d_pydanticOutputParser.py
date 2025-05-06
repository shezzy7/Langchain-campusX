"""
 PydanticOutputParser is a structured output parser in langchain that uses pydantic models to enforce schema validation when processing LLM responses.
  It ensures that LLM response follow a well defined structure,it automatically converts LLM outputs into python objects,it uses python's built-in validation to catch incorrect or missing data, and it works well with other langchain components.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    google_api_key=gemini_api_key
)

# first of all we will build a class in which we will set constraints for each attribute and then will pass this class to out PydanticOutputParser for setting our validations on data

class Person(BaseModel):
    name: str = Field(description='Name of personalities described in output')
    age: int = Field(gt=18,description="Age of personalities")
    net_Worth : int = Field(description='net worth of each person')
    bussines : list[str] = Field(description='list out all the bussinesses that he has')
    location: list[int] = Field(description='Loaction of personalities in the form of lang,long ')


parser = PydanticOutputParser(pydantic_object=Person) #PydanticOutputParser takes an object as input(which is our class name)
template = PromptTemplate(
    template = """
        give me data of 5 famous {domain} \n {formate_instruction}    
    """,
    input_variables=['domain'],
    partial_variables={'formate_instruction' : parser.get_format_instructions()}
)
# print(template.invoke({'domain':'tech giants'})) #By printing this we can see our prompt which goes to llm.
chain = template | model | parser
response = chain.invoke({'domain':'businessmen'})
print(response)