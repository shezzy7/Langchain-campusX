
    
# from langchain_google_genai import ChatGoogleGenerativeAI   #ChatGoogleGenerativeAI is used when we want to use chat model instead of a llm
# from langchain_core.messages import HumanMessage
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()
# import os
# st.header("Research Tool")
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# llm = ChatGoogleGenerativeAI(
#     model='gemini-2.0-flash-exp-image-generation',
#     google_api_key=gemini_api_key
# )
# user_input = st.text_input("Enter prompt here")
# if st.button("Summarize"):
#     result=llm.invoke(user_input)
#     st.write(result.content)
    
# result = llm.invoke([
#     HumanMessage(content="Waht is ihram in islam?")
# ])

from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
st.header("Research tool")
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp-image-generation",
    google_api_key=gemini_api_key
)
user_input = st.text_input("Enter prompt here..")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)