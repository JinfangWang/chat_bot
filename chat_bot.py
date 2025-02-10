import streamlit as st
import openai
import pinecone
import torch
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from pinecone import Pinecone

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone
if not pinecone_api_key:
    raise ValueError("⚠️ PINECONE_API_KEY is not set! Please check your environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# List all indexes to verify your index exists
index_name = "datascience-bot-index"  # Change this to your actual index name
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found! Ensure you created it in Pinecone.")

# Connect to the existing Pinecone index
index = pc.Index(index_name)

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define CSV file for storing chat history
CHAT_LOG_FILE = "chat_history.csv"

# Ensure CSV file exists
if not os.path.exists(CHAT_LOG_FILE):
    df = pd.DataFrame(columns=["user_query", "bot_response"])
    df.to_csv(CHAT_LOG_FILE, index=False)

# Streamlit UI
st.title("📚 Chatbot for Introductory Data Science")
st.write("Ask your questions and get relevant responses based on course materials!")

# Chat history stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_query = st.chat_input("Ask your question here...")

if user_query:
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate embedding for user query
    query_embedding = embedding_model.encode(user_query).tolist()

    # Retrieve top relevant documents from Pinecone
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Combine retrieved text
    retrieved_texts = [match["metadata"]["text"] for match in search_results["matches"]]
    context = "\n\n".join(retrieved_texts)

    # Call OpenAI GPT-4 model
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering students' questions based on the retrieved documents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer concisely."}
        ],
        temperature=0.5
    )

    bot_response = response.choices[0].message.content.strip()

    # Store the chat in CSV
    df = pd.read_csv(CHAT_LOG_FILE)
    new_data = pd.DataFrame([[user_query, bot_response]], columns=["user_query", "bot_response"])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(CHAT_LOG_FILE, index=False)

    # Display bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)