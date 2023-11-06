import weaviate
import os
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_client_secret = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
    },
    # auth_client_secret=auth_client_secret
)

details = client.schema.get()

print("The classes inside your Weaviate cluster are:")
for index ,iter in enumerate(details['classes']):
    print(f"{index}. {iter['class']}")
