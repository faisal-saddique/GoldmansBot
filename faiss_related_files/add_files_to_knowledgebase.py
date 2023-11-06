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
from langchain.vectorstores import FAISS  # For creating a vector database for similarity search
from langchain.embeddings.openai import OpenAIEmbeddings  # For generating embeddings with OpenAI's embedding model
from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# Load environment variables from .env file
load_dotenv()

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

print(f"Pushing docs to FAISS...")

# Create OpenAIEmbeddings object using the provided API key
embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_documents(docs, embeddings)

print("Docs pushed to FAISS.")

print("Saving Index locally.")

# Check if the folder super_knowledgebase already exists
if os.path.exists("super_knowledgebase"):
    print("Merging with an existing index...")
    existing_docsearch = FAISS.load_local(folder_path="super_knowledgebase", embeddings=embeddings, index_name="main")
    existing_docsearch.merge_from(docsearch)
    existing_docsearch.save_local("super_knowledgebase", index_name="main")
    print("Merged index saved.")
else:
    docsearch.save_local("super_knowledgebase", index_name="main")
    print("New index saved.")