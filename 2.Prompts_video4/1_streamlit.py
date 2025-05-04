
from langchain_google_genai import ChatGoogleGenerativeAI   #ChatGoogleGenerativeAI is used when we want to use chat model instead of a llm
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
st.header("Research Paper Summarizer")
gemini_api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp-image-generation',
    google_api_key=gemini_api_key
)

article_name = st.selectbox("Choose an article",[
    "Kashf al-Mahjub","Mathnawi","Fusus al-Hikam","Futuhal Al-Ghaib","Ar-Risalah al-Qushayriyya","Ihya Ulum al-Din"
])

exp_style = st.selectbox("Select explanation style" , [
    "Beginner friendly","Technical","Code-Oriented"
])

exp_length = st.selectbox("Choose length",[
    "Short  ( 1-2 Paragraph)","Medium  (3-5 Paragraph)","Long (Detailed Explnation)"
])

# we can use f string instead of prompt template but it has many benefits like it is compatible with langchain ecosystem,we can store a template in a json file and can use it where we want in as many files as needed.It also gives error during runtime if any of the variables is not given or misspelled.
# template = PromptTemplate(
#     template="""
#     Please summarize the research paper titled "{article_name}" with the following specifications:
#     Explanation Style : {exp_style}
#     EXplanation length : {exp_length}
#     1.Mathematical Details:
#         - Include relevent mathematial equations if present in paper.
#         - Explain the mathematical concepts using simple,intuitive code snippets where applicable.
#     2.Analogies:
#         - Use relateable analogies to simplify complex ideas.
#     If certain information is not available in the paper,respond with: "Insufficient information available" instead of guessing      
#     """,
#     input_variables=['article_name',"exp_style","exp_length"]
# )

template =  load_prompt("template.json") #this will get template in a correct way and will be stored in referring variable.
# prompt = template.invoke({
#     'article_name':article_name,
#     'exp_style':exp_style,
#     'exp_length':exp_length
# })
# here we can also short our code by using chaining concept where we will be calling invoke method just once instead of calling it separately for model and template we will be doing so in following way
chain = template | llm
# here this chain will call invoke method first for template by passing given arguemnts and then will call invoke method for llm by passing prompt returned by invoke of template
result = chain.invoke(
    {
        'article_name':article_name,
        'exp_style':exp_style,
        'exp_length':exp_length
    }
)
if st.button("Summarize"):
    # here we don't need to call invoke again for llm as it is already called by using concept of chain
    # result=llm.invoke(prompt)
    st.write(result.content)
    
