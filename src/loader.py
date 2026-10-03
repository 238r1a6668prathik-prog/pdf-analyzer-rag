from langchain_community.document_loaders import PyPDFLoader


# Give the path of the PDF file
pdf_path = "data/Python_Notes.pdf"


# Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()


# Check whether the PDF was loaded successfully
print("PDF loaded successfully!")
print("Number of pages:", len(documents))


# Display some content from the first page
print("\nFirst page content:\n")
print(documents[0].page_content[:1000])