import streamlit as st
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# Заголовок приложения
st.title("🤖 Тренинг-агент")
st.write("Задавайте вопросы на основе загруженного кейса!")

# Инициализация векторного хранилища
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return vectordb

vectordb = load_vector_db()

# Настройка LLM
llm = OpenAI(temperature=0)

# Создаем цепочку для QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# Ввод вопроса от пользователя
question = st.text_input("Введите ваш вопрос:")

if question:
    with st.spinner("Генерация ответа..."):
        response = qa_chain.invoke({
            "query": f"Отвечай строго на основе кейса. Если информации нет, скажи 'В кейсе это не указано'. Вопрос: {question}"
        })

        answer = response["result"]
        sources = response.get("source_documents", [])

        st.subheader("Ответ")
        st.write(answer)

        if sources:
            st.subheader("Источники")
            for doc in sources:
                st.write(f"- {doc.metadata.get('source', 'Неизвестно')}")
