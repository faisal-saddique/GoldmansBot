# Import required libraries
from langchain.document_loaders import (
    PyMuPDFLoader,  # For loading PDF files
    DirectoryLoader,  # For loading files from a directory
    TextLoader,  # For loading plain text files
    Docx2txtLoader,  # For loading DOCX files
    UnstructuredPowerPointLoader,  # For loading PPTX files
    UnstructuredExcelLoader  # For loading XLSX files
)

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader  # For loading CSV files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into smaller chunks
from dotenv import load_dotenv  # For loading environment variables from .env file
import os
import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

# Load environment variables from .env file
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

# Replace with the name of the directory carrying your data
data_directory = "data"


# Load your documents from different sources
def get_documents():
    print("Loading documents...")

    # Create loaders for PDF, text, CSV, DOCX, PPTX, XLSX files in the specified directory
    pdf_loader = DirectoryLoader(data_directory, glob="**/*.pdf", loader_cls=PyMuPDFLoader,
                                show_progress=True, use_multithreading=True, silent_errors=True)
    text_loader_kwargs = {'autodetect_encoding': True}
    txt_loader = DirectoryLoader(data_directory, glob="**/*.txt", loader_cls=TextLoader, show_progress=True,
                                use_multithreading=True, silent_errors=True, loader_kwargs=text_loader_kwargs)
    csv_loader = DirectoryLoader(data_directory, glob="**/*.csv", loader_cls=CSVLoader,
                                show_progress=True, use_multithreading=True, silent_errors=True)
    docx_loader = DirectoryLoader(data_directory, glob="**/*.docx", loader_cls=Docx2txtLoader,
                                show_progress=True, use_multithreading=True, silent_errors=True)
    pptx_loader = DirectoryLoader(data_directory, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader,
                                show_progress=True, use_multithreading=True, silent_errors=True)
    xlsx_loader = DirectoryLoader(data_directory, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader,
                                show_progress=True, use_multithreading=True, silent_errors=True)

    # Initialize the 'docs' variable
    docs = None

    # Load files using the respective loaders
    pdf_data = pdf_loader.load()
    txt_data = txt_loader.load()
    csv_data = csv_loader.load()
    docx_data = docx_loader.load()
    pptx_data = pptx_loader.load()
    xlsx_data = xlsx_loader.load()

    # Combine all loaded data into a single list
    docs = pdf_data + txt_data + docx_data + pptx_data + xlsx_data

    print(f"Total {len(docs)} documents loaded.")
    
    # Return all loaded data
    if csv_data:
        return docs, csv_data
    else:
        return docs, None

# Get the raw documents from different sources
raw_docs, csv_docs = get_documents()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

docs = text_splitter.split_documents(raw_docs)

if csv_docs:
    docs = docs + csv_docs

print(f"Total chunks: {len(docs)}")

print(f"Pushing docs to Weaviate...")

retriever = WeaviateHybridSearchRetriever(
    client=client,
    index_name="PerryDocs",
    text_key="text",
    attributes=[],
    create_schema_if_missing=True,
)

chunk_size = 1000
total_docs = len(docs)

for start in range(0, total_docs, chunk_size):
    end = min(start + chunk_size, total_docs)
    chunk = docs[start:end]
    
    # Print some information about the current chunk
    print(f"Pushing chunk {start // chunk_size + 1} of {total_docs // chunk_size + 1}")
    print(f"Chunk size: {len(chunk)} documents")
    
    retriever.add_documents(chunk)

    # Print a message after each chunk is pushed
    print(f"Chunk {start // chunk_size + 1} pushed successfully.")

print("All docs pushed to Weaviate.")
