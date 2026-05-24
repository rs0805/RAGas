from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from load_data import load_chunks

load_dotenv(find_dotenv())

def build_retriever():
    chunks = load_chunks()
    print(f"Chunks loaded: {len(chunks)}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
    )
    return retriever