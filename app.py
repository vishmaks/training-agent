import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.question_answering import load_qa_chain

# Cache the vector database loading
@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load vector database
vectordb = load_vector_db()

# Initialize the LLM
llm = OpenAI(temperature=0)

# UI
st.title("Case-based Q&A Agent")

question = st.text_input("Enter your question based on the case:")

if question:
    # Search for similar documents
    docs = vectordb.similarity_search(question, k=3)
    
    # Create QA chain
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    
    # Get answer
    answer = qa_chain.run(input_documents=docs, question=question)
    
    st.write("### Answer:")
    st.write(answer)
