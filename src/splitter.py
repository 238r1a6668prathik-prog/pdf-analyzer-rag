from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Give the path of the PDF file
pdf_path = "data/Python_Notes.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("Number of pages:", len(documents))


# Set the chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Split the PDF into smaller chunks
chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# Display the first chunk
print("\nFirst chunk:\n")
print(chunks[0].page_content)


# Display information about the first chunk
print("\nMetadata:")
print(chunks[0].metadata)