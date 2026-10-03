from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Load the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Load the saved FAISS vector database
vector_store = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded successfully!")


# Create a retriever to find relevant chunks
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# Question that we want to search in the PDF
question = "What is Python?"


# Find the three most relevant chunks
results = retriever.invoke(question)


# Display the retrieved chunks
print("\nNumber of relevant chunks:", len(results))

for i, document in enumerate(results):

    print("\n" + "=" * 60)
    print(f"RESULT {i + 1}")
    print("=" * 60)

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)