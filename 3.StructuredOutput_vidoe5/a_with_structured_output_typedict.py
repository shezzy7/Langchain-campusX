from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel 
from typing import Annotated,Optional,Literal
import os

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GEMINI_API_KEY
)

# we can generate responses from models in our own defined structure.As we know when we give any query to our model then it generates response in plain text but we can define a formate for output structure and gave it to our model telling him that what kind of output we want(json formate etc).

# here we are doing a basic practical example of it.We are creating a class which is inherited by BaseModel and inside this we are defining the structure of desired output.We are saying that output should contain two attributes one is summary of type string and other is sentiment also of type str.Output type can be changed by model and system will not give any error if type of any attribute is changed but the structure will remain same as defined here.

class Review(BaseModel):
    summary:str
    sentiment:str

# there exists a model's method named 'with_structured_output' and we pass name of class inside this method for defining structure of output.And this fucntion call generates a copy of this model which generates response in given formate and we invoke model through this copy.
structured_model = model.with_structured_output(Review)

# response = structured_model.invoke("he is a good guy.he went to school from age 4.He is a bussiness man now and is earning a good income.He also pays taxes and also is hardworker for achieving his dreams.He likes animals")
# print(response)

# here in above class's structure model is understanding that from attributes that he has to make summary and sentiment of given input.But sometime if we are giving complex attribute names then we can also define that this attributes is specified for which thing(basically we can attach a nested prompt with each attribute.).For this purpose we use Annotated method imported from typing

class Review2(BaseModel):
    summary:Annotated[str,"Give brief summary of given input"]
    # we can also use literal for giving llm specific items form which he has to chose one.
    sentiment:Annotated[Literal['neg','pos','neutral'],"Make a sentiment about the given input"]
    key_themes:Annotated[list[str],"Highlight the key themes used in given input"]
    #  we can define any attribute as optional that if any thing related to this attribute is present then tell as leave it as its optional.For this purpose we use optional method imported from typing
    pros:Annotated[Optional[list[str]] ,"write any pros given about anything given in user input"]

structured_model2=model.with_structured_output(Review2)
response2=structured_model2.invoke("The newly released action movie, 'Cosmic Fury,' has been met with generally positive reviews. Critics praised its stunning visual effects and high-octane action sequences, with many highlighting the impressive CGI dinosaurs. However, some viewers found the plot to be somewhat predictable and lacking in character development. Despite this, the film's strong emphasis on thrilling prehistoric creature encounters and its overall energetic pace made it an enjoyable cinematic experience for many. Several reviews specifically mentioned the breathtaking scenes involving the velociraptors and the final, epic battle with a massive theropod. While the narrative depth could have been improved, the sheer spectacle and excitement were undeniable")
print(response2)

