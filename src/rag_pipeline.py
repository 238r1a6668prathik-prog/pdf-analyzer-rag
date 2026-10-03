from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS


# Load the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Load the FAISS vector database
vector_store = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded successfully!")


# Create a retriever to find relevant PDF chunks
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# Load the Qwen model for generating answers
llm = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0
)


# Question we want to ask the PDF
question = "What is Python?"


# Find the most relevant chunks for the question
documents = retriever.invoke(question)


# Combine the retrieved chunks into one context
context = "\n\n".join(
    document.page_content
    for document in documents
)


# Create a prompt using the retrieved PDF content
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


# Send the prompt to Qwen and generate the answer
response = llm.invoke(prompt)


# Display the generated answer
print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)


# Display the PDF pages used for the answer
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