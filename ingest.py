import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = []


for file in os.listdir("datasets"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"datasets/{file}")
        pdf_docs = loader.load()

        for d in pdf_docs:
            d.metadata["source"] = file

        docs.extend(pdf_docs)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)


vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

vectorstore.persist()
print("Ingestion complete")
