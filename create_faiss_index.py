import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load documents from the "docs" folder
def load_documents():
    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()

# Main
if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("❌ No documents found in the 'docs' folder.")
        exit(1)

    print(f"✅ Loaded {len(documents)} documents. Creating FAISS index...")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local("faiss_index")
    print("✅ FAISS index created and saved successfully!")
