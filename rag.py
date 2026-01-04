from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


llm = OllamaLLM(
    model="mistral",
    temperature=0
)

def rag_answer(question, source=None):
    if source:
        docs = vectorstore.similarity_search(
            question, k=3, filter={"source": source}
        )
    else:
        docs = vectorstore.similarity_search(question, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt)

if __name__ == "__main__":
    print(
        rag_answer("How gen AI gains insight from data?")
    )
