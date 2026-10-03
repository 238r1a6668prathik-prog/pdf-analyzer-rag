from langchain_ollama import OllamaEmbeddings


# Load the embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Text used to test the embedding model
text = "This is a test document for our PDF Analyzer RAG project."


# Convert the text into a vector
vector = embeddings.embed_query(text)


# Display the embedding details
print("Embedding created successfully!")
print("Number of dimensions:", len(vector))
print("First 5 values:", vector[:5])