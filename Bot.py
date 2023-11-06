# Import necessary libraries
import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
import weaviate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain.prompts import (HumanMessagePromptTemplate,SystemMessagePromptTemplate,ChatPromptTemplate)

from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# Get environment variables
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
auth_client_secret = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

# Set Streamlit page configuration
st.set_page_config(page_title="Jaarvis", page_icon="🔥")
st.title("🔥 Jaarvis")

# Function to configure the retriever and load the index
@st.cache_resource(ttl="1h")
def configure_retriever():
    # Create a Weaviate client
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={
            "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
        },
        # auth_client_secret=auth_client_secret
    )

    # Configure the WeaviateHybridSearchRetriever
    retriever = WeaviateHybridSearchRetriever(
        client=client,
        index_name="PerryDocs",
        text_key="text",
        k=6,
        attributes=["source", "page"],
        create_schema_if_missing=False,
    )

    llm = ChatOpenAI(temperature=0)

    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    # return retriever_from_llm

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_from_llm)

    return compression_retriever

# Callback handler for streaming text
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

# Callback handler to print retrieval results
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            # source = os.path.basename(doc.metadata["file_path"])
            self.status.write(f"**Document: {idx}**") #  | {source} | {doc.metadata['page']}
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

def scrape_website_recursively(url):
    with st.sidebar:
        loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text, exclude_dirs=["https://www.dreamworldvision.com/a/dreamtopia/meta-quest-3-a-comprehensive-overview","https://www.dreamworldvision.com/a/dreamtopia/hololens-a-dim-outlook-or-a-bright-future","https://www.dreamworldvision.com/a/dreamtopia/augmented-reality-the-future-of-retail-and-marketing","https://www.dreamworldvision.com/a/dreamtopia/apple-vision-pro-a-game-changer-or-a-gimmick"])
        docs = loader.load()
            # Create OpenAIEmbeddings object using the provided API key
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_documents(docs, embeddings)
        # Check if the folder super_knowledgebase already exists
        if os.path.exists("super_knowledgebase"):
            st.info("Merging with an existing index...")
            existing_docsearch = FAISS.load_local(folder_path="super_knowledgebase", embeddings=embeddings, index_name="main")
            existing_docsearch.merge_from(docsearch)
            existing_docsearch.save_local("super_knowledgebase", index_name="main")
            st.info("Merged index saved.")
        else:
            docsearch.save_local("super_knowledgebase", index_name="main")
            st.info("New index saved.")
        st.cache_resource.clear()

# Add a sidebar to ask for a URL
url = st.sidebar.text_input("Enter a URL")
if st.sidebar.button("Submit"):
    # Call your function with the entered URL
    if url:
        scrape_website_recursively(url)

# Configure the retriever
retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, input_key='question', output_key='answer')

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True
)

# Define the system message template
system_template = """End every answer by asking the user if he needs more help. Use the following pieces of context to answer the users question. 
If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
----------------
{context}"""

# Create the chat prompt templates
messages = [
SystemMessagePromptTemplate.from_template(system_template),
HumanMessagePromptTemplate.from_template("{question}")
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, return_source_documents=True, combine_docs_chain_kwargs={"prompt": qa_prompt}
)

# Display existing chat history
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Accept user input and provide responses
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain(user_query, callbacks=[retrieval_handler, stream_handler])