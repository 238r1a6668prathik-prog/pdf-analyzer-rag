import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS


# Set up the Streamlit page
st.set_page_config(
    page_title="PDF Analyzer using RAG",
    page_icon="📄",
    layout="wide"
)


# Display the application title
st.title("📄 PDF Analyzer using RAG")

st.write(
    "Upload a PDF and ask questions based on its content."
)


# Load the embedding model and LLM
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0
)


# Allow the user to upload a PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# Save and process the uploaded PDF
if uploaded_file is not None:

    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"PDF uploaded: {uploaded_file.name}"
    )


    if st.button("🔄 Process PDF"):

        with st.spinner("Processing PDF..."):

            # Read the PDF and extract its text
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()


            # Break the extracted text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(
                documents
            )


            # Convert the chunks into embeddings and store them in FAISS
            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )


            # Save the FAISS index for later searches
            vector_store.save_local(
                "vectorstore"
            )


        st.success(
            f"PDF processed successfully! "
            f"{len(documents)} pages and "
            f"{len(chunks)} chunks created."
        )


# Get the question from the user
question = st.text_input(
    "Ask a question about your PDF:"
)


# Search the PDF and generate an answer
if st.button("🔍 Ask"):

    if not question:
        st.warning("Please enter a question.")

    elif not os.path.exists(
        "vectorstore/index.faiss"
    ):
        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        with st.spinner("Finding answer..."):

            # Load the saved FAISS vector database
            vector_store = FAISS.load_local(
                "vectorstore",
                embeddings,
                allow_dangerous_deserialization=True
            )


            # Use the vector database to find relevant chunks
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 3}
            )


            # Retrieve the most relevant chunks for the question
            documents = retriever.invoke(
                question
            )


            # Combine the retrieved chunks to create the context
            context = "\n\n".join(
                document.page_content
                for document in documents
            )


            # Give the retrieved context and question to the LLM
            prompt = f"""
You are a PDF Analyzer.

Answer the question using ONLY the
information provided in the PDF context.

Do not use outside knowledge.

If the answer is not present in the
context, say:

"I could not find the answer in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


            # Generate the final answer
            response = llm.invoke(prompt)


        # Show the generated answer
        st.subheader("💡 Answer")

        st.write(response.content)


        # Show the pages used to generate the answer
        st.subheader("📚 Sources")

        seen_pages = set()

        for document in documents:

            page = document.metadata.get(
                "page_label"
            )

            if page is None:
                page = document.metadata.get(
                    "page", 0
                ) + 1

            source = document.metadata.get(
                "source",
                uploaded_file.name
            )

            source_name = os.path.basename(
                source
            )

            source_key = (
                source_name,
                page
            )

            if source_key not in seen_pages:

                st.write(
                    f"📄 {source_name} — "
                    f"Page {page}"
                )

                seen_pages.add(
                    source_key
                )