from langchain_core.prompts import ChatPromptTemplate

# we can also make dynamic prompt templates.In which input variables are inserted dynamically.
chat_template = ChatPromptTemplate(
    [
        ("system","Your are a helpfull {domain} expert"),
        ("user","explain in simple terms, what is {topic}")
    ]
    
    
)
prompt = chat_template.invoke(
    {'domain':"cricket",'topic':"over"}
)
print(prompt)

# A MessagesPlaceholder in langchain is a special placeholder inside ChatPromptTemplate to dynamically insert chat history or a list of messages at runtime.