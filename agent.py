import streamlit as st
import tempfile
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_together import Together
from langchain_community.embeddings import HuggingFaceEmbeddings

# Настройки страницы
st.set_page_config(page_title="Агент для тренинга", page_icon="🤖")

st.title("🤖 Агент для тренинга (БЕСПЛАТНО через TogetherAI)")
st.write("Загрузите файл (PDF, DOCX, TXT), и агент будет отвечать на вопросы, основываясь только на содержании кейса.")

# Ввод API-ключа TogetherAI
api_key = st.text_input("Введите TogetherAI API Key", type="password")

if api_key:
    uploaded_file = st.file_uploader("📂 Загрузите файл кейса", type=["pdf", "docx", "txt"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        documents = loader.load()

        # Разбиваем текст
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Создаём эмбеддинги (через HuggingFace – бесплатно)
        embeddings = HuggingFaceEmbeddings()
        vectordb = Chroma.from_documents(texts, embeddings)

        # LLM через TogetherAI
        llm = Together(
            together_api_key=api_key,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.2,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
            chain_type="stuff"
        )

        st.success("✅ Кейс загружен! Теперь можно задавать вопросы.")
        question = st.text_input("Введите ваш вопрос по кейсу")

        if question:
            answer = qa_chain.run(
                f"Отвечай строго на основе кейса. Если информации нет, скажи 'В кейсе это не указано'. Вопрос: {question}"
            )
            st.markdown(f"### 💡 Ответ:\n{answer}")

    else:
        st.info("📥 Загрузите файл (PDF, DOCX или TXT), чтобы начать.")
else:
    st.warning("Введите API-ключ TogetherAI для продолжения.")
