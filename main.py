import os
import streamlit as st
import pickle
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from dotenv import load_dotenv
load_dotenv()  # load environment variables (like OPENAI_API_KEY)

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("📰 News Research Tool")
st.sidebar.title("Controls")

file_path = "vector_index.pkl"
llm = OpenAI(temperature=0.9, max_tokens=500)

# Sidebar option to rebuild index if needed
rebuild_index = st.sidebar.checkbox("Rebuild Vector Index (if new data added)", value=False)

# Placeholder
main_placeholder = st.empty()

# ----------------------------
# Load or Rebuild Index
# ----------------------------
if rebuild_index:
    main_placeholder.text("⚙️ Rebuilding Vector Index from local articles...")
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    # Load local articles
    loader = DirectoryLoader("articles/", glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    data = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    # Create embeddings and FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    main_placeholder.text("✅ Vector Index rebuilt and saved!")

# ----------------------------
# Load existing index
# ----------------------------
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
    main_placeholder.text("📂 Vector Index loaded successfully.")
else:
    main_placeholder.text("❌ No vector index found. Please rebuild it first.")

# ----------------------------
# User Query
# ----------------------------
query = st.text_input("🔎 Ask a question about your articles:")
if query and os.path.exists(file_path):
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)

    # Show Answer
    st.header("📝 Answer")
    st.write(result["answer"])

    # Show Sources
    sources = result.get("sources", "")
    if sources:
        st.subheader("📌 Sources")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)
