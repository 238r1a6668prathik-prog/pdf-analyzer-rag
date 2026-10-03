from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load PDF
pdf_path = "data/Python_Notes.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Number of pages:", len(documents))


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# Display first chunk
print("\nFirst chunk:\n")
print(chunks[0].page_content)


# Display metadata
print("\nMetadata:")
print(chunks[0].metadata)