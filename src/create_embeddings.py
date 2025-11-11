"""
create_embeddings.py
--------------------
Generates embeddings for the Ministry of Education AI Library System.
Implements Smart Chunking (semantic text splitting) and Metadata Filtering.
Creates FAISS vector index combining Books and User Behavior datasets.
"""

import os
import pandas as pd
from pathlib import Path
 # --- Smart import for all LangChain versions ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "❌ Could not find RecursiveCharacterTextSplitter. "
            "Try running: pip install -U langchain-text-splitters"
        )

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
try:
    # ✅ For modern LangChain versions (>=0.1+)
    from langchain_core.documents import Document
except ImportError:
    # ✅ Backward compatibility for older versions
    from langchain.docstore.document import Document

from dotenv import load_dotenv

# 1️⃣ Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file.")

# 2️⃣ Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectors"
VECTOR_DIR.mkdir(exist_ok=True)

books_path = DATA_DIR / "books_dataset.csv"
users_path = DATA_DIR / "users_behavior.csv"

# 3️⃣ Smart Chunking configuration
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # each chunk ~800 tokens
    chunk_overlap=120,   # overlap ensures context continuity
    separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""],
)

def load_docs_from_csv(path: Path, doc_type: str):
    """
    Convert CSV rows into LangChain Document objects with metadata.
    - doc_type: 'books' or 'users'
    """
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return []

    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        # Concatenate all text columns
        text = " ".join([str(v) for v in row.values if isinstance(v, str) and v.strip()])

        # Apply smart chunking
        chunks = splitter.split_text(text)

        for chunk in chunks:
            metadata = {
                "source": doc_type,
                "id": row.get("book_id", row.get("user_id", "N/A")),
                "subject": row.get("subject", "unknown"),
                "language": row.get("language", "en"),
                "year": row.get("year", "unknown"),
            }
            docs.append(Document(page_content=chunk, metadata=metadata))

    print(f"✅ Loaded {len(docs)} chunks from {path.name}")
    return docs


# 4️⃣ Load & process both datasets
print("📘 Loading library datasets...")
books_docs = load_docs_from_csv(books_path, "books")
users_docs = load_docs_from_csv(users_path, "users")

all_docs = books_docs + users_docs
print(f"📚 Total combined chunks to embed: {len(all_docs)}")

if not all_docs:
    raise ValueError("❌ No documents found to embed. Check CSV files in data/.")

# 5️⃣ Initialize embeddings & FAISS index
print("⚙️  Generating embeddings using OpenAI API...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

vectorstore = FAISS.from_documents(all_docs, embeddings)

# 6️⃣ Save local FAISS index
faiss_path = VECTOR_DIR / "faiss_index"
vectorstore.save_local(str(faiss_path))
print(f"🎉 Embeddings successfully created and saved at: {faiss_path}")

