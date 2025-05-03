from langchain_google_genai import GoogleGenerativeAI   #GoogleGenrativeAI is used when we are refering to llms instead of chat models

from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")
# print(gemini_api_key)
llm = GoogleGenerativeAI(
    model="models/gemini-2.0-flash-exp-image-generation",
    google_api_key = gemini_api_key
    
) 

prompt = input("What you want to ask? -> ")

content = f"Give me answer in 50 words : {prompt}"

result = llm.invoke(content)
print(result)