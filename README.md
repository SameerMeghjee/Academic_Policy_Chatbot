# Academic_Policy_Chatbot
# 🎓 Academic Policy Chatbot

This is a Retrieval-Augmented Generation (RAG) based chatbot that allows students to ask questions about their **Academic Policy Manual**. It uses a pre-uploaded PDF document, processes it into vector embeddings, and answers questions using the powerful LLaMA 3.3 70B model from Groq.

---

## 🚀 Features

- Chatbot interface via Streamlit
- Automatically reads and embeds an academic policy PDF
- Retrieves relevant context using semantic search (FAISS + MiniLM)
- Answers student queries using Groq LLaMA 3.3
- No Groq API key required in frontend (hidden in backend)

---

## 📦 Dependencies

Install dependencies using:
pip install -r requirements.txt

## 🧠 Architecture
User Query ➜ Semantic Search[FAISS + MiniLM] ➜ Extract Relevant Chunks[via Groq API] ➜ LLaMA 3.3 ➜ Answer

▶️ How to Run
- Clone this repository:

git clone https://github.com/your-username/academic-policy-chatbot.git

cd academic-policy-chatbot

- Install requirements:

pip install -r requirements.txt

- Add your Groq API Key in app.py:

GROQ_API_KEY = "your-groq-api-key"

- Place your/any PDF file as academic_policy.pdf in the root folder.

- Run the app:

streamlit run app.py
