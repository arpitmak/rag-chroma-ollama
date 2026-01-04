from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

def retrieve(query, source=None, k=3):
    if source:
        return vectorstore.similarity_search(
            query, k=k, filter={"source": source}
        )
    return vectorstore.similarity_search(query, k=k)

if __name__ == "__main__":
    docs = retrieve("Market and competitive intel")
    for d in docs:
        print(d.page_content)
