import os
import streamlit as st
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai  # ✅ Google AI Studio

st.set_page_config(page_title="Iqra University Academic Policy Chatbot",
                   page_icon="🎓", layout="wide")

# === Load environment variables ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Add your Gemini key in .env
PDF_PATH = "academic_policy.pdf"

# === Configure Gemini API ===
genai.configure(api_key=GOOGLE_API_KEY)

# === Top Layout ===
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    pass

with col2:
    logo_url = "https://profiles.pk/wp-content/uploads/2018/02/iqrauniversitylogo.jpg"
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; align-items: flex-end; height: 120px;'>
            <img src="{logo_url}" alt="Logo" style="width: 120px;" />
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    pass

# === Page Title ===
st.markdown("<h2 style='text-align: center;'>🎓 Academic Policy Chatbot</h2>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 17px;'>Ask any question about your student policies below 👇</p>",
            unsafe_allow_html=True)
st.divider()

# === Load PDF and create vectorstore ===
@st.cache_resource(show_spinner="🔄 Loading academic policy...")
def prepare_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

vectorstore = prepare_vectorstore(PDF_PATH)

# === Retrieve relevant context from the document ===
def retrieve_context(query, vectorstore, top_k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

# === Generate response using Google Gemini ===
def generate_answer(query, context):
    prompt = (
        f"You are a knowledgeable assistant. Use the academic policy context below "
        f"to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error from Google API: {e}"

# === Initialize session state ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Display chat history ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Accept new user input ===
query = st.chat_input("Ask something about the academic policy...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("✍️ Thinking..."):
            context = retrieve_context(query, vectorstore)
            answer = generate_answer(query, context)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
