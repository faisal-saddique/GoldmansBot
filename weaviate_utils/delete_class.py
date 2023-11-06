import weaviate
import os
from dotenv import load_dotenv

load_dotenv()

class_name = "MandeepDocsTry"

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_client_secret = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
    },
    auth_client_secret=auth_client_secret
)

class_details = client.schema.delete_class(class_name=class_name)

print(f"The class {class_name} was deleted successfully!")