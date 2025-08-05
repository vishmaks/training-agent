import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Set page config
st.set_page_config(page_title="Case Assistant", layout="wide")

# Load FAISS index
@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

st.title("📄 Case Assistant")
st.write("Ask questions based on the uploaded case file. If the answer is not found, the assistant will say: 'This is not mentioned in the case.'")

# Load vector DB
vectordb = load_vector_db()

# Create retriever
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize QA chain
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# User input
question = st.text_input("Enter your question in English:")

if st.button("Get Answer"):
    if question:
        result = qa_chain.invoke({"query": f"Answer strictly based on the case file. If the information is missing, respond with: 'This is not mentioned in the case.' Question: {question}"})
        answer = result.get("result", "No answer found.")
        st.write("### Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
