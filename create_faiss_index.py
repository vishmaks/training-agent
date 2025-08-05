import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def load_documents():
    pdf_path = "docs"
    documents = []

    for file_name in os.listdir(pdf_path):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_path, file_name))
            documents.extend(loader.load())
    return documents

if __name__ == "__main__":
    print("Loading PDF documents...")
    documents = load_documents()

    if not documents:
        print("❌ No PDF documents found in the 'docs' folder.")
        exit(1)

    print(f"✅ Loaded {len(documents)} pages from PDFs. Creating FAISS index...")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")

    print("✅ FAISS index created and saved successfully!")