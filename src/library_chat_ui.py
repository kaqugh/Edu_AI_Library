import streamlit as st
from src.rag_pipeline import run_rag

st.set_page_config(page_title="Edu AI Library Chat", page_icon="📚", layout="wide")

st.title("📚 Edu AI Library Assistant")
st.caption("Ask questions about the library — powered by RAG + OpenAI Embeddings")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about books, students, or topics..."):
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = run_rag(prompt)
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
        st.markdown(response)

    # Save response
    st.session_state["messages"].append({"role": "assistant", "content": response})
