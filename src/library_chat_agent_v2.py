"""
library_chat_agent_v2.py
========================
Advanced version of the AI Library Chat Agent.
Automatically selects between Books and Users indexes based on user intent,
and remembers conversation context.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from pathlib import Path

# 1️⃣ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found. Please check your .env file.")
    st.stop()

# 2️⃣ Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORS_DIR = BASE_DIR / "vectors"

# 3️⃣ Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# Helper: Load FAISS index
def load_index(name):
    path = VECTORS_DIR / name
    if path.exists():
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    else:
        st.warning(f"⚠️ {name} index not found.")
        return None

books_index = load_index("books_faiss")
users_index = load_index("users_faiss")

if not books_index and not users_index:
    st.error("❌ No FAISS indexes found. Please run create_embeddings.py first.")
    st.stop()

# 4️⃣ Initialize model + memory
chat_model = ChatOpenAI(
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4-turbo"
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5️⃣ Function: classify query intent (Book vs User)
def classify_intent(query: str) -> str:
    query_lower = query.lower()
    if any(word in query_lower for word in ["student", "teacher", "borrow", "user", "behavior", "member", "id", "u0"]):
        return "users"
    return "books"

# 6️⃣ Dynamic retrieval logic
def get_retriever_for_query(query: str):
    intent = classify_intent(query)
    if intent == "books" and books_index:
        return books_index.as_retriever(search_kwargs={"k": 5})
    elif intent == "users" and users_index:
        return users_index.as_retriever(search_kwargs={"k": 5})
    else:
        return books_index.as_retriever(search_kwargs={"k": 3})

# 7️⃣ Create Streamlit UI
st.set_page_config(page_title="AI Library Assistant v2", page_icon="📚")
st.title("📖 AI Library Chat Assistant v2")
st.caption("Ask about books, authors, subjects, or user borrowing behaviors.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("💬 Ask your question:")

if user_query:
    with st.spinner("Thinking..."):
        retriever = get_retriever_for_query(user_query)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory
        )
        response = qa_chain({"question": user_query})
        answer = response["answer"]

        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("AI", answer))

# 8️⃣ Display chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑‍💻 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")

st.sidebar.header("⚙️ Chat Memory")
if st.sidebar.button("🗑️ Clear Memory"):
    st.session_state.chat_history = []
    memory.clear()
    st.sidebar.success("Memory cleared successfully!")
