from langchain_ollama import OllamaEmbeddings
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
# 4. Ask a question
# -----------------------------

question = "What is the main objective of this document?"


# -----------------------------
# 5. Retrieve relevant chunks
# -----------------------------

results = retriever.invoke(question)


# -----------------------------
# 6. Display results
# -----------------------------

print("\nNumber of relevant chunks:", len(results))

for i, document in enumerate(results):

    print("\n" + "=" * 60)
    print(f"RESULT {i + 1}")
    print("=" * 60)

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)