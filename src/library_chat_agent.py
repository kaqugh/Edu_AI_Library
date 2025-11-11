"""
library_chat_agent.py
=====================
Interactive Streamlit chatbot for the AI Library System.
Queries FAISS vector stores (books + users) and uses OpenAI GPT model
to provide answers based on semantic search results.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pathlib import Path

# 1️⃣ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found. Please check your .env file.")
    st.stop()

# 2️⃣ Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORS_DIR = BASE_DIR / "vectors"

# 3️⃣ Initialize embeddings and load indexes
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def load_faiss_index(name):
    path = VECTORS_DIR / name
    if path.exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    else:
        st.warning(f"⚠️ {name} index not found.")
        return None

books_index = load_faiss_index("books_faiss")
users_index = load_faiss_index("users_faiss")

# 4️⃣ Create the conversational chain (combine indexes if both exist)
retrievers = []
if books_index:
    retrievers.append(books_index.as_retriever(search_kwargs={"k": 5}))
if users_index:
    retrievers.append(users_index.as_retriever(search_kwargs={"k": 5}))

if not retrievers:
    st.error("❌ No vector indexes found. Please run create_embeddings.py first.")
    st.stop()

# Use the first retriever by default (you can merge later)
retriever = retrievers[0]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_model = ChatOpenAI(
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4-turbo"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory
)

# 5️⃣ Streamlit UI
st.set_page_config(page_title="AI Library Assistant", page_icon="📚")
st.title("📖 AI Library Chat Assistant")
st.caption("Ask about books, subjects, authors, or user behaviors.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("💬 Ask your question:")

if user_query:
    with st.spinner("Thinking..."):
        response = qa_chain({"question": user_query})
        answer = response["answer"]
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("AI", answer))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑‍💻 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
