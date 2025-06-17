import os
import streamlit as st
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Add your Groq API key in .env
PDF_PATH = "academic_policy.pdf"

# === Top bar with logo and student image ===
col1, col2 = st.columns([1, 3])

with col1:
    logo_path = "https://profiles.pk/wp-content/uploads/2018/02/iqrauniversitylogo.jpg"  
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    else:
        st.markdown("**[Logo Missing]** Upload a logo.png")

with col2:
    student_image_path = "https://img.freepik.com/premium-photo/portrait-student-holding-books-library_357704-1410.jpg"  # Replace with your student image filename
    if os.path.exists(student_image_path):
        st.image(student_image_path, width=250, use_column_width=False)
    else:
        st.markdown("**[Student Image Missing]** Upload students.png")

# === Streamlit page config ===
st.set_page_config(page_title="Iqra University Academic Policy Chatbot", page_icon="🎓", layout="wide")
st.markdown("<h2 style='text-align: center;'>🎓 Academic Policy Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 17px;'>Ask any question about your student policies below 👇</p>", unsafe_allow_html=True)
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

# === Generate response using Groq's LLaMA 3 ===
def generate_answer(query, context):
    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        f"You are a knowledgeable assistant. Use the academic policy context below to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Groq's optimized LLaMA 3 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant for answering academic policy questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Error from Groq API: {e}"

# === Initialize session state ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Display chat messages ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Accept new input ===
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
