import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Load and split the document into chunks
def load_and_split(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# Build the vector database
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
    vectordb.persist()
    return vectordb

# Create a RetrievalQA chain for question answering
def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
