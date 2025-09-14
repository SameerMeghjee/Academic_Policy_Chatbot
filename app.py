import streamlit as st
from groq import Groq
import os

# ─────────────────────────────────────────────
# ✅ Set your Groq API key
#    (recommended to set as environment variable or Streamlit secret)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# Streamlit App UI
# ─────────────────────────────────────────────
st.title("Academic Policy Chatbot 📚")
st.write("Ask me anything about academic policies and I’ll answer using Groq LLaMA-3 models.")

# User Input
user_query = st.text_area("Enter your question:")

if st.button("Get Answer"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # ─────────────────────────────────────────────
                # ✅ Groq Chat Completion
                # Supported Models (Sept 2025):
                #   • llama-3-70b   (best quality)
                #   • llama-3-8b    (smaller, cheaper)
                #   • mixtral-8x7b  (fast MoE)
                #   • gemma-7b      (cost efficient)
                # ─────────────────────────────────────────────
                response = client.chat.completions.create(
                    model="llama-3-70b",  # ✅ current stable model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for academic policy questions."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.3
                )
                answer = response.choices[0].message.content
                st.success(answer)

            except Exception as e:
                st.error(f"⚠️ Groq API Error: {str(e)}")
