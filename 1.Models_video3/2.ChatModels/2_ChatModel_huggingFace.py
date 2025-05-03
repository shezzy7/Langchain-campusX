from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
huggingFace_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/QwQ-32B",
    task="text-generation",
    huggingfacehub_api_token=huggingFace_api_key
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is capital of USA?")
print(result.content)