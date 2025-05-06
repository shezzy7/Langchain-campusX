# StructuredOutputParser is asn output parser in Langchain that helps extract structured JSON data from LLM responses based on predefined field schemas.
# It works by defining a list of fields(ResponseSchema) that the model should return,ensuring the output follows a structured format.

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    google_api_key=gemini_api_key
)

# we can generate output in json formate by setting our own key names instead of relying on model that he will generate key according to data using JsonOutputParser as we do earlier in previous file b_jsonOutputParser.py

# fisrst of all we will define our scheme using ResponseSchema
schema = [
    # in ResponseSchema we have to pass two things one is name of key and other is the descrition about this key like-> ResponseSchema(name='nameOfKey' , description="description about this key")
    # lets suppose here we want to get three facts about a topic and we want that each fact should be wraped inside a key
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description="Fact 3 about the topic"),
    ResponseSchema(name='names', description='Names of persons if any in topic given')
]

# then we have to send this schema to StructuredOutputParser's from_response_schemas method and store the returned value into a variable.
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='''
    Give me three facts about {topic} \n {formate_instruction}
    ''',
    input_variables=['topic'],
    # here parser has stored value returned by structuredOutputParser into a function and calling this function we get formate instructions and we pass it here.
    partial_variables={'formate_instruction':parser.get_format_instructions()}
)

# without chaining
prompt = template.invoke({'topic':'Prophet Muhammad'})
general_response = model.invoke(prompt)
final_response = parser.parse(general_response.content)#after getting response we have to parse it first 
print(final_response)


# lets do this through chaining
chain = template | model | parser
response = chain.invoke({'topic':"Top highly earning businesses in the world"})
print(response)

# StructuredOutputParser has some disadvantages like we can't validate data type of any string here.For example if we want to get name,age and location of someone in data.Then our model can tell that he 25 years old which is string data type by what if we want this as integer.In this case out StructuredOutputParser does not work.To overcome this problem we use PydanticOutputParser