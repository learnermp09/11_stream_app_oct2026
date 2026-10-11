from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1. Load PDFs, split text, create embeddings, and save FAISS
def load_data_into_vectorstore():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    if not documents:
        raise ValueError("No PDF pages found. Check the ./data folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_db")

    return "FAISS database created successfully."


# 2. Load FAISS and create the LangChain chain
def create_qa_chain():
    embeddings = HuggingFaceEmbeddings()

    db = FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever()

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.01
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer using only the context below.
If the answer is not in the context, say:
Not present in the documents provided!!!

Context:
{context}

Question:
{question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Compose the Runnable chain
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


# 4. Basic Streamlit interface
st.title("PDF Question Answering with LangChain")

if st.button("Load PDFs into FAISS"):
    try:
        with st.spinner("Loading PDFs and creating FAISS..."):
            message = load_data_into_vectorstore()
        st.success(message)
    except Exception as error:
        st.error(f"Error: {error}")


question = st.text_input("Ask a question about the PDF")

if st.button("Ask") and question.strip():
    try:
        qa_chain = create_qa_chain()

        # 5. Stream the answer as chunks arrive
        st.write_stream(
            qa_chain.stream(question.strip())
        )

    except Exception as error:
        st.error(f"Error: {error}")
        st.caption(
            "Check your GROQ_API_KEY, FAISS database, "
            "and selected Groq model."
        )