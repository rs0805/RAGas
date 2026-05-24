from datasets import Dataset
from rag_chain import create_rag_chain

def build_dataset():
    rag_chain, retriever = create_rag_chain()
    questions = [
        "What did the president say about Justice Breyer?",
        "What did the president say about Intel's CEO?",
        "What did the president say about gun violence?",
    ]
    ground_truths = [
        [
            "The president said that Justice Breyer dedicated his life to serving the country and thanked him for his service."
        ],
        [
            "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."
        ],
        [
            "The president asked Congress to pass proven measures to reduce gun violence."
        ],
    ]
    answers = []
    contexts = []
    for query in questions:
        answer = rag_chain.invoke(query)
        retrieved_docs = retriever.invoke(query)
        context = [
            doc.page_content
            for doc in retrieved_docs
        ]
        answers.append(answer)
        contexts.append(context)
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [gt[0] for gt in ground_truths],
    }
    dataset = Dataset.from_dict(data)
    return dataset