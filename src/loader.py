from langchain_community.document_loaders import PyPDFLoader


pdf_path = "data/Python_Notes.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("PDF loaded successfully!")
print("Number of pages:", len(documents))

print("\nFirst page content:\n")
print(documents[0].page_content[:1000])