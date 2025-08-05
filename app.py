from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Load the vector database
@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectordb = load_vector_db()

# Configure LLM with the correct model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Create Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff"
)

# Streamlit UI
st.title("Training Agent")
query = st.text_input("Enter your question:")
if query:
    result = qa_chain.invoke({"query": query})
    st.write(result["result"])
