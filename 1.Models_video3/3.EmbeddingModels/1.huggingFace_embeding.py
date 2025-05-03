

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

import os
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    

)

text = [
    
    "Islamabad is the capital of Pakistan.",
    "New Delhi is the capital of India",
    "Washington is the capital of USA",
    "Kolkata is the capital of BAN",
    "Abu Dahabi is the capital of UAE"
    ]
# we use embed_documents when we pass a list of input
vector = llm.embed_documents(text)
print(str(vector))