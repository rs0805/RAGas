import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

URL = "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"

def load_chunks():
    response = requests.get(URL)
    with open("state_of_the_union.txt", "w", encoding="utf-8") as file:
        file.write(response.text)
    loader = TextLoader(
    "state_of_the_union.txt",
    encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")
    return chunks