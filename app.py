import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Заголовок страницы
st.set_page_config(page_title="Training Agent", layout="wide")
st.title("🤖 Training Agent")
st.write("Задавайте вопросы по загруженному кейсу. Если информации нет, агент ответит: 'В кейсе это не указано'.")

# Функция загрузки базы векторного поиска
@st.cache_resource
def load_vector_db():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Инициализация векторной базы
try:
    vectordb = load_vector_db()
except Exception as e:
    st.error("Ошибка при загрузке базы данных. Сначала запустите скрипт create_faiss_index.py локально и закоммитьте faiss_index в репозиторий.")
    st.stop()

# Инициализация LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Создаем цепочку вопросов-ответов
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=False
)

# Интерфейс Streamlit
question = st.text_input("Введите ваш вопрос:")

if st.button("Задать вопрос"):
    if question.strip():
        with st.spinner("Ищу ответ..."):
            response = qa_chain.invoke({"query": f"Отвечай строго на основе кейса. Если информации нет, скажи 'В кейсе это не указано'. Вопрос: {question}"})
            st.write("### Ответ:")
            st.write(response["result"])
    else:
        st.warning("Введите вопрос перед отправкой.")
