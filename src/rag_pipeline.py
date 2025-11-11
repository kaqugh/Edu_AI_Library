import os
import sqlite3
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.semantic_filters import infer_metadata_from_query

# === CONFIGURATION ===
DATA_PATH = Path.home() / "Edu_AI_Library" / "data"
VECTOR_PATH = Path.home() / "Edu_AI_Library" / "vectors" / "faiss_index"
CACHE_PATH = Path.home() / "Edu_AI_Library" / ".cache"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

# === SEMANTIC CACHE SETUP ===
DB_PATH = CACHE_PATH / "answers.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS cache (
        query TEXT PRIMARY KEY,
        answer TEXT
    )"""
)
conn.commit()
print(f"⚡ Semantic cache: SQLite at {DB_PATH}")

# === LOAD EMBEDDINGS + VECTOR STORE ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local(str(VECTOR_PATH), embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

print("RAG pipeline ready. Example calls:\n")
print("• run_rag('List Arabic history books for grade 10')")
print("• run_rag('What subjects does student U015 borrow most?', {'source':'users'})")


# === CACHE HANDLER ===
def get_cached_answer(query):
    cursor.execute("SELECT answer FROM cache WHERE query=?", (query,))
    result = cursor.fetchone()
    return json.loads(result[0]) if result else None


def store_cached_answer(query, answer):
    cursor.execute(
        "INSERT OR REPLACE INTO cache (query, answer) VALUES (?, ?)",
        (query, json.dumps(answer)),
    )
    conn.commit()


# === MAIN RAG FUNCTION ===
def run_rag(query: str, filters: dict = None, use_cache: bool = True):
    print(f"\n🔎 Query: {query}")

    # 1️⃣ Check cache
    if use_cache:
        cached = get_cached_answer(query)
        if cached:
            print("💾 Cache hit (SQLite)")
            return cached

    # 2️⃣ Infer semantic filters if none provided
    if filters is None:
        inferred = infer_metadata_from_query(query)
        print(f"🧠 Inferred filters: {inferred}")
        filters = inferred
    else:
        print(f"📘 Using provided filters: {filters}")

    # 3️⃣ Retrieve initial candidate docs
    docs = retriever.invoke(query)
    if not docs:
        print("⚠️ No documents found by retriever.")
        return []

    # 4️⃣ Apply metadata-based filtering
    filtered_docs = []
    for d in docs:
        meta = d.metadata
        if all(
            str(meta.get(k, "")).lower() == str(v).lower()
            for k, v in filters.items()
            if k in meta
        ):
            filtered_docs.append(d)

    if not filtered_docs:
        print("⚠️ No relevant items found after applying semantic metadata filters.")
        print("💡 Try adjusting filters (subject/language/year) or updating dataset.")
        return []

    print(f"✅ Retrieved {len(filtered_docs)} relevant items after filtering.")

    # 5️⃣ Prepare final response (basic summary)
    results = []
    for d in filtered_docs:
        snippet = d.page_content[:200].replace("\n", " ")
        results.append({
            "title": d.metadata.get("title", "Unknown"),
            "subject": d.metadata.get("subject", "N/A"),
            "language": d.metadata.get("language", "N/A"),
            "text": snippet
        })

    # 6️⃣ Store in semantic cache
    if use_cache:
        store_cached_answer(query, results)

    return results


# === AGENTIC DEMO ===
if __name__ == "__main__":
    print("\nRunning sample semantic query...\n")
    results = run_rag("List Arabic history books for grade 10")
    for r in results:
        print(f"📘 {r['title']} ({r['language']}) — {r['subject']}")

