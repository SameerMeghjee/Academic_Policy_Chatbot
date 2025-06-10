import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
import tempfile

# === Configuration ===
GROQ_API_KEY = "" #Add your Groq API key here
PDF_PATH = "academic_policy.pdf"  

# === Page Settings ===
st.set_page_config(page_title="Academic Policy Chatbot", page_icon="🎓", layout="wide")
st.markdown("<h2 style='text-align: center;'>🎓 Academic Policy Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 17px;'>Ask any question about your student policies below 👇</p>", unsafe_allow_html=True)
st.divider()

# === Load and process the pre-uploaded PDF ===
@st.cache_resource(show_spinner="🔄 Loading academic policy...")
def prepare_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = prepare_vectorstore(PDF_PATH)

# === Chat Functions ===
def retrieve_context(query, vectorstore, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

def generate_answer(query, context, groq_client):
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer the question using only the context provided above."
    )
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering academic policy questions."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# === Session state for chat ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === User Input ===
query = st.chat_input("Ask something about the academic policy...")

# === Display chat messages ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Process new question ===
if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("✍️ Thinking..."):
            context = retrieve_context(query, vectorstore)
            groq_client = Groq(api_key=GROQ_API_KEY)
            answer = generate_answer(query, context, groq_client)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
