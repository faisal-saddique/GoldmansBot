import weaviate
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

class_name = "JimmyDocs"

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_client_secret = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
    },
    auth_client_secret=auth_client_secret
)

class_details = client.schema.get(class_name=class_name)

print(f"The class details for class {class_name} are:")
pprint(class_details)
