from langchain.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
print(OpenAI().predict("Hello"))