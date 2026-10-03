import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS


# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="PDF Analyzer using RAG",
    page_icon="📄",
    layout="wide"
)


# -----------------------------------
# Title
# -----------------------------------

st.title("📄 PDF Analyzer using RAG")

st.write(
    "Upload a PDF and ask questions based on its content."
)


# -----------------------------------
# Initialize models
# -----------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="qwen2.5-coder:3b",
    temperature=0
)


# -----------------------------------
# Upload PDF
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# -----------------------------------
# Process PDF
# -----------------------------------

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

            # Load PDF
            loader = PyPDFLoader(pdf_path)

            documents = loader.load()


            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(
                documents
            )


            # Create FAISS vector store
            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )


            # Save vector store
            vector_store.save_local(
                "vectorstore"
            )


        st.success(
            f"PDF processed successfully! "
            f"{len(documents)} pages and "
            f"{len(chunks)} chunks created."
        )


# -----------------------------------
# Question input
# -----------------------------------

question = st.text_input(
    "Ask a question about your PDF:"
)


# -----------------------------------
# Ask button
# -----------------------------------

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

            # Load FAISS
            vector_store = FAISS.load_local(
                "vectorstore",
                embeddings,
                allow_dangerous_deserialization=True
            )


            # Retriever
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 3}
            )


            # Retrieve documents
            documents = retriever.invoke(
                question
            )


            # Create context
            context = "\n\n".join(
                document.page_content
                for document in documents
            )


            # Prompt
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


            # Generate answer
            response = llm.invoke(prompt)


        # -----------------------------------
        # Display answer
        # -----------------------------------

        st.subheader("💡 Answer")

        st.write(response.content)


        # -----------------------------------
        # Display sources
        # -----------------------------------

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