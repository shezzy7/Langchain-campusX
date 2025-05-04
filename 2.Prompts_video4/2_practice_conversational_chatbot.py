# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-exp-image-generation",
#     google_api_key=GEMINI_API_KEY
# )

# # this will keep on running untill user enters 'exit' as a query.
# # while True:
# #     user_input = input("You: ")
# #     if user_input=='exit':
# #         break
# #     result = llm.invoke(user_input)
# #     print("AI: "+result.content)

# # But the problem with above code is that our model does not stores history so we cannot ask anything relevant to our previous questions.To solve this  problem we can use a list for storing our messages in it and then send this list to model as a prompt and model will understand itself that about what thing we are talking about and he will answer it.

# # messages=[]

# # while True:
# #     user_input = input("You: ")
# #     if user_input=='exit':
# #         break
# #     messages.append(user_input)
# #     result = llm.invoke(messages)
# #     messages.append(result.content)
# #     print("AI: "+result.content)

# # But this is also problematic because each message is not clarified that which message is from user and which is from AI.Because our list only contains messages not the info about source.
# # So langchain have solved this problem by introducing some keywords like HumanMessage,AIMessage,SystemMessage
# # SystemMessage is like a instruction to mode.Eg->"You are a helpfull assistant and give me correct answer in a polite way."
# # HummanMessage is the message/query sended by the user to the model.Eg.->"What is the capital of Pakistan?"
# #AIMessage is the response sended by the Model.Eg.->"The capital of Pakistan is Islamabad"

# # First we have to import them
# from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
# messages = [
#     SystemMessage(content="You are a helpful assistant")
# ]

# import streamlit as st
# st.header("Conversational Chatbot")

# i=0
# while True:
#     user_input=st.text_input("You : " , key=i)
#     i+=1
#     messages.append(HumanMessage(content=user_input))
#     if user_input=='exit':
#         break
#     response = llm.invoke(messages)
#     messages.append(AIMessage(response.content))
#     st.write("AI : "+response.content)

# print(messages)


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Recommended for chat, or your preferred model
    google_api_key=GEMINI_API_KEY
)

# Streamlit UI
st.title("🧠 Gemini Chatbot")
st.write("Ask anything. Type 'exit' to end.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]

# User input
user_input = st.text_input("You:", key="user_input")

if user_input:
    if user_input.lower() == "exit":
        st.write("Goodbye 👋")
    else:
        # Append human message
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Get response from LLM
        response = llm.invoke(st.session_state.messages)

        # Append AI response to history
        st.session_state.messages.append(AIMessage(content=response.content))

# Display the chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**AI:** {msg.content}")
