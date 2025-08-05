import streamlit as st
from agent import load_documents, create_vectorstore, create_qa_chain

st.title("🤖 Training Agent (OpenAI)")

# Загрузка PDF
uploaded_file = st.file_uploader("Загрузите PDF с кейсом", type="pdf")

if uploaded_file:
    with open("case.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Кейс загружен!")
    st.write("Строим базу знаний...")

    docs = load_documents("case.pdf")
    vectordb = create_vectorstore(docs)
    qa_chain = create_qa_chain(vectordb)

    st.success("База знаний готова!")

    question = st.text_input("Введите вопрос по кейсу:")

    if question:
        with st.spinner("Генерируем ответ..."):
            answer = qa_chain.run(f"Отвечай строго на основе кейса. Если информации нет, скажи 'В кейсе это не указано'. Вопрос: {question}")
            st.write("### Ответ:")
            st.write(answer)
