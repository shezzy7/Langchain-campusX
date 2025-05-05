# Pydantic is a data validation and data parsing library for python.It ensures that the data you work with is correct,structured and type-safe.
# With its help we can set default values in attributes,can set optional fields,can apply constraints on attributes etc.
# Filed function in pydantic is used to apply contraints,validate an email using EmailStr

from pydantic import BaseModel,EmailStr,Field
from typing import Optional,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    google_api_key=GEMINI_API_KEY
)
class Student(BaseModel):
    name:str
    age:Optional[int]=Field(default=None)
    # Field method of pydantic help us in applying constraints,default values and also we can add a description about any atribute for model understanding.
    # her we have applied contraints that value of cgpa should be greater than 0 and should be smaller than 10.And also we can set default value here.
    cgpa:float = Field(gt=0,lt=10,default=5,description='A deciaml value representing the cgpa of the students')
    email:EmailStr
    program:Annotated[str,"Degree which the student is doing"]
    sentiment:str
    


structured_model = model.with_structured_output(Student)
str='Shahzad hussain is student of computer science.He was born at 13 april,2000.His acdemic is vip and now his current cgpa is 3.39.He is one of the tall guys in class.He looks good.He wants to do MS in usa.You can contact him through email->shahzadhussain72242@gmail.cm.He is trying to be good,to stay good.May Allah help him!'

json_formate = {
  "characteristics": {
    "type": "object",
    "properties": {
      "look": {
        "type": "string",
        "description": "Describes the visual appearance.",
        "default": "Nyc"
      },
      "name": {
        "type": "string",
        "description": "The name of the entity.",
        "required": 'true'
      }
    },
    "required": [
      "look",
      "name"
    ]
  }
}

structured_modell = model.with_structured_output(json_formate)
response = structured_modell.invoke(str)
print(response)

# from pydantic import BaseModel, EmailStr, Field
# from typing import Optional, Annotated

# class Student(BaseModel):
#     name: str
#     age: Optional[int] = None  # Explicitly set default to None for Optional
#     cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the students')
#     email: Optional[EmailStr] = None  # Uncommented and made Optional with default None
#     program: Annotated[str, "Degree which the student is doing"]

# # Corrected dictionary keys and value types
# stu_obj = Student(name="shahzad", cgpa=7.5, email="shezzy@gmail.com", program="cs")
# print(stu_obj)

# # Example with missing optional age
# # stu_obj_no_age = Student(name="ali", cgpa=9.1, email="ali@example.com", program="it")
# # print(stu_obj_no_age)