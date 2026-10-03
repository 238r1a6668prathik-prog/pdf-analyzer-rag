from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS


# -----------------------------
# 1. Load embedding model
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -----------------------------
# 2. Load FAISS vector store
# -----------------------------

vector_store = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded successfully!")


# -----------------------------
# 3. Create retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------
# 4. Load Qwen
# -----------------------------

llm = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0
)


# -----------------------------
# 5. User question
# -----------------------------

question = "What is Python?"


# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------

documents = retriever.invoke(question)


# -----------------------------
# 7. Create context
# -----------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# -----------------------------
# 8. Prompt
# -----------------------------

prompt = f"""
You are a PDF Analyzer.

Answer the question using ONLY the information
provided in the PDF context.

Do not use outside knowledge.

If the answer is not present in the context,
say: "I could not find the answer in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# 9. Generate answer
# -----------------------------

response = llm.invoke(prompt)


# -----------------------------
# 10. Display answer
# -----------------------------

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)


# -----------------------------
# 11. Display sources
# -----------------------------

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

seen_pages = set()

for document in documents:

    page = document.metadata.get("page_label")

    if page is None:
        page = document.metadata.get("page", 0) + 1

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    source_name = source.split("\\")[-1].split("/")[-1]

    source_key = (source_name, page)

    if source_key not in seen_pages:
        print(f"📄 {source_name} — Page {page}")
        seen_pages.add(source_key)