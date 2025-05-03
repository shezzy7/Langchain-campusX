from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

scholars = [
    "Raza Saqib Mustafai – A popular Pakistani Islamic scholar known for his emotional and spiritual speeches.",
    "Tariq Jamil – A globally recognized Islamic preacher promoting peace, unity, and moral reform.",
    "Saqib Shami – A Pakistani Islamic speaker known for his love-focused sermons about the Prophet (PBUH).",
    "Taqi Usmani – A leading Islamic jurist and expert in Islamic finance and Shariah law.",
    "Khadim Hussain Rizvi – The late founder of TLP known for his fiery speeches and defense of blasphemy laws."
    "Illyas Attari - He is the founder of Dawat-e-Islami."
]

query = "Tell me about dawat e islami"
# when we have a single string then we use embed_query for generating its vectors.
query_embed=embeding_model.embed_query(query)
# and use embed_documents when we have a list of inputs to convert into vectors.
docs_embed = embeding_model.embed_documents(scholars)
# cosine similarity method takes two vector 2d arrays as arguments and returns the value of similarity of first array with second array's each item in a 2d array
similarity = cosine_similarity([query_embed],docs_embed)[0] #cosine_similarity method returns a 2d array,but inside this array there is only one element(which is also a 1d array) so we are getting only that one particular element by using index 0.
index,item = sorted(list(enumerate(similarity)),key=lambda x:x[1])[-1] #here enumerate method will combine index of each value with it.And lambda key is basically being used for sorting the given list on the basis of second element.As we know after sorting our main value(value with large similarity wwill be at last we are getting it by using index -1)

print(scholars[index])

