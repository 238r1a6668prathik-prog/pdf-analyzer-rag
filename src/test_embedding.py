from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

text = "This is a test document for our PDF Analyzer RAG project."

vector = embeddings.embed_query(text)

print("Embedding created successfully!")
print("Number of dimensions:", len(vector))
print("First 5 values:", vector[:5])