from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task='text-generation',
    pipeline_kwargs=dict(
        temprature=0.5,
        max_new_tokens=1000
    ),
    HUGGINGFACEHUB_API_TOKEN=hf_token
    
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is capital of Pakistan?")

print(result)