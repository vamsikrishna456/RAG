import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from groq import Groq
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_available_groq_models(api_key):
    default_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "allam-2-7b"
    ]
    if not api_key:
        return default_models
    try:
        client = Groq(api_key=api_key)
        models_data = client.models.list()
        chat_models = [
            m.id for m in models_data.data
            if not any(x in m.id.lower() for x in ["whisper", "guard", "orpheus"])
        ]
        return sorted(chat_models) if chat_models else default_models
    except Exception:
        return default_models


def get_documents_text(docs):
    text = ""
    for doc in docs:
        try:
            if doc.name.lower().endswith(".pdf"):
                pdf_reader = PdfReader(doc)
                if getattr(pdf_reader, "is_encrypted", False):
                    try:
                        pdf_reader.decrypt("")
                    except Exception:
                        pass
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            elif doc.name.lower().endswith(".txt"):
                text += doc.read().decode("utf-8", errors="ignore") + "\n"
        except Exception as e:
            st.error(f"Error reading {doc.name}: {e}")
    return text


get_pdfs_text = get_documents_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(raw_text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def get_session_history(session_id: str):
    if "chat_store" not in st.session_state:
        st.session_state.chat_store = {}

    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.chat_store[session_id]


def get_conversation_chain(vector_store, api_key=None, model_name="openai/gpt-oss-20b"):
    llm = ChatGroq(
        groq_api_key=api_key or os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=0.3
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant that answers questions
based on the provided document context.

Use the context below to answer the user's question.

If the answer cannot be found in the provided context,
clearly say that you don't know based on the provided documents.

Context:

{context}
"""
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),
        (
            "human",
            "{input}"
        )
    ])

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    def rag_function(inputs):
        docs = retriever.invoke(
            inputs["input"]
        )

        formatted_context = format_docs(docs)

        messages = prompt.invoke({
            "context": formatted_context,
            "chat_history": inputs.get(
                "chat_history",
                []
            ),
            "input": inputs["input"]
        })

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "context": docs
        }

    rag_chain = RunnableLambda(
        rag_function
    )

    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversation_chain


def main():
    load_dotenv()

    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="📚"
    )

    st.title("📚 RAG Chatbot")
    st.caption("Ask questions about your PDF documents using Groq")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask a question about your documents..."
    )

    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        if st.session_state.conversation:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation.invoke(
                            {"input": user_question},
                            config={
                                "configurable": {
                                    "session_id": "default_user"
                                }
                            }
                        )

                        answer = response["answer"]
                        st.markdown(answer)

                        if response.get("context"):
                            with st.expander("🔍 View Retrieved Sources"):
                                for i, doc in enumerate(response["context"], 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.caption(doc.page_content)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })
                    except Exception as e:
                        err_text = f"Groq API Error: {e}"
                        st.error(err_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ {err_text}"
                        })

        else:
            with st.chat_message("assistant"):
                st.warning(
                    "Please upload and process documents first."
                )

    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Enter your Groq API key (or set GROQ_API_KEY in .env)"
        )

        effective_key = groq_api_key.strip() or os.getenv("GROQ_API_KEY", "").strip()
        available_models = get_available_groq_models(effective_key)
        default_index = (
            available_models.index("openai/gpt-oss-20b")
            if "openai/gpt-oss-20b" in available_models
            else 0
        )

        groq_model = st.selectbox(
            "Groq Model",
            options=available_models,
            index=default_index,
            help="Available models for your Groq account"
        )

        st.divider()
        st.header("📄 Documents")

        docs = st.file_uploader(
            "Upload your documents (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button(
            "Process Documents",
            use_container_width=True
        ):
            if not docs:
                st.warning(
                    "Please upload at least one PDF or TXT file."
                )
                return

            effective_key = groq_api_key.strip() or os.getenv("GROQ_API_KEY", "").strip()
            if not effective_key:
                st.error(
                    "Please enter a Groq API Key above or add it to your .env file."
                )
                return

            with st.spinner("Processing documents..."):
                raw_text = get_documents_text(
                    docs
                )

                if not raw_text.strip():
                    st.error(
                        "No readable text could be extracted from the uploaded document(s). Please verify the file contents."
                    )
                    return

                text_chunks = get_text_chunks(
                    raw_text
                )

                vector_store = get_vectorstore(
                    text_chunks
                )

                st.session_state.conversation = (
                    get_conversation_chain(
                        vector_store,
                        api_key=effective_key,
                        model_name=groq_model
                    )
                )

                st.session_state.messages = []

                st.success(
                    "Documents processed successfully! You can now ask questions."
                )


if __name__ == "__main__":
    main()

