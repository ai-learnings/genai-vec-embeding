import json
import os
from typing import List

import voyageai
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_ollama import OllamaEmbeddings

# Load environment variables from .env file
load_dotenv()

# Connect to your Atlas cluster
uri = os.getenv("MONGODB_URI")
mongoClient = MongoClient(uri, server_api=ServerApi('1'))
mongoCollection = mongoClient[os.getenv("MONGODB_DATABASE")][os.getenv("MONGODB_COLLECTION")]


# To generate document embedings
# embedingModelName = os.getenv("VOYAGE_MODEL")
# voClient = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

embedingModelName = os.getenv("OLLAMA_MODEL")
ollama = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model=embedingModelName
)


# # function that returns embedings from voyage-3.5
# def listOfEmbededVectorsVoyage(string_list):
#     print("--------------Vyoage vector embading-------------------")
#     result = []
#     embedingModalName = "voyage-3.5"
#     result = voClient.embed(string_list, model=embedingModelName)
#     print("embeding done using -> "+embedingModalName)
#     return result.embeddings



# function that returns embedings using ollama -> mxbai-embed-large
def listOfEmbededVectorsOllama(input_texts: list):
    vectors = ollama.embed_documents(input_texts)
    return vectors



class BaseEmbedingEntityLLM:
    def __init__(self, data: str, data_embeded: List[float], modal_name):
        self.data: str = data
        self.data_embeded: List[float] = data_embeded
        self.embeding_modal: str = modal_name



def connectToMongoDB():
    # Send a ping to confirm a successful connection
    try:
        mongoClient.admin.command('ping')
        print("Pinged your mongo atlas deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)



##############################################################################################################################

# check mongo connection.
# generate vector embeddings.
# save vector embadings in monog datatabse.

if __name__ == "__main__":
    connectToMongoDB()
    
    data = ["the president of dhaked firm is laxmi devi.",
            "praveen brother name is umesh.",
            "The owner of dhaked-firm is praveen."]
    
    embeded_data: List[List[float]] = listOfEmbededVectorsOllama(data)
    
    entities : List[BaseEmbedingEntityLLM] = []
    for index,ed in enumerate(embeded_data):
        entity = BaseEmbedingEntityLLM(data[index],ed,embedingModelName)  
        entities.append(entity)
        
    result = mongoCollection.insert_many([ent.__dict__ for ent in entities])
    result._raise_if_unacknowledged
    print("Inserted acknowledged:", result.acknowledged)
        
        