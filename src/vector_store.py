from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Load the PDF and extract its content
pdf_path = "data/Python_Notes.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("PDF loaded successfully!")
print("Number of pages:", len(documents))


# Split the PDF text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# Create embeddings for the document chunks
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Creating embeddings...")


# Store the embeddings in a FAISS vector database
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created successfully!")


# Save the vector database locally
vector_store.save_local("vectorstore")

print("Vector store saved successfully!")