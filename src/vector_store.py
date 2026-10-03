from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# 1. Load PDF
# -----------------------------

pdf_path = "data/Python_Notes.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("PDF loaded successfully!")
print("Number of pages:", len(documents))


# -----------------------------
# 2. Split PDF into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Creating embeddings...")


# -----------------------------
# 4. Create FAISS vector store
# -----------------------------

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created successfully!")


# -----------------------------
# 5. Save vector store
# -----------------------------

vector_store.save_local("vectorstore")

print("Vector store saved successfully!")