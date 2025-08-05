import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load FAISS vector database
@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize Streamlit app
st.title("Case Assistant")
st.write("Ask a question based on the case documents. The assistant will provide **detailed** answers.")

# User input
question = st.text_input("Enter your question in English:")

if question:
    vectordb = load_vector_db()

    # Configure LLM with more detailed response generation
    llm = OpenAI(
        temperature=0.7,       # Slightly creative but still grounded
        max_tokens=1200,       # Longer, more detailed answers
        model_name="gpt-4"     # Use GPT-4 if available, otherwise fallback to GPT-3.5
    )

    # Setup QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),  # Retrieve more context
        return_source_documents=True
    )

    # Create prompt for detailed answers
    query = f"""
    Provide a clear and detailed answer to the following question.
    Use only the information from the case.
    If the case does not contain the answer, explicitly state: "This is not specified in the case."
    Add explanations where possible.

    Question: {question}
    """

    # Run query
    result = qa_chain.invoke({"query": query})

    # Display the result
    st.subheader("Answer:")
    st.write(result["result"])

    # Show source documents for transparency
    with st.expander("Show source documents"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Document {i+1}:** {doc.metadata.get('source', 'Unknown')}")
            st.write(doc.page_content)
