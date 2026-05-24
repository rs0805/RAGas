from langchain_openai import ChatOpenAI  
from langchain_core.prompts import ChatPromptTemplate          # changed
from langchain_core.runnables import RunnablePassthrough       # changed
from langchain_core.output_parsers import StrOutputParser

from vector_store import build_retriever

def create_rag_chain():
    retriever = build_retriever()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    template = """
    You are an assistant for question-answering tasks.
    Use the following retrieved context to answer the question.
    If you don't know the answer, say you don't know.
    Use two sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever