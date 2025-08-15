import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Setup ---
st.set_page_config(page_title="Case Chatbot", page_icon="💬")
st.title("💬 Case-based Chatbot")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set.")
    st.stop()

@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectordb = load_vector_db()

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    max_tokens=800
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
    chain_type="refine"
)

# --- Chat interface ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask your question about the case...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = qa_chain.run(f"Strictly answer based on the case. If not in case, say: 'Not mentioned in the case'. Question: {user_input}")
        st.markdown(response)

    st.session_state.chat_history.append((user_input, response))
