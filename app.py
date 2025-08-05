import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA

# Кешируем загрузку векторной базы
@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Загружаем базу
vectordb = load_vector_db()

# Создаём цепочку для поиска ответов
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# Интерфейс Streamlit
st.title("💬 Тренинг-агент")
question = st.text_input("Введите вопрос по кейсу:")

if st.button("Получить ответ"):
    if question.strip():
        result = qa_chain.invoke(
            {"query": f"Отвечай строго на основе кейса. Если информации нет, скажи 'В кейсе это не указано'. Вопрос: {question}"}
        )
        st.write("### Ответ:")
        st.write(result["result"])
    else:
        st.warning("Введите вопрос!")
