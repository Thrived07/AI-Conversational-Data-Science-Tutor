import streamlit as st
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# Set up the Streamlit UI
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("📚 AI Conversational Data Science Tutor")
st.write("Ask your data science-related queries, and I will help you out!")

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = "Enter your api key"

# Ask for API key only if not already stored
if not st.session_state.api_key:
    api_key = st.text_input("Enter your Google AI API Key:", type="password")
    if api_key:
        st.session_state.api_key = api_key  # Store in session state
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key saved! You won't need to enter it again.")

# Proceed only if API key is set
if st.session_state.api_key:
    # Initialize memory for conversation awareness
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

    # Initialize the Gemini 1.5 Pro Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=st.session_state.api_key)

    # Create a conversational chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory
    )

    # User input section
    user_input = st.text_input("Enter your data science question:", "")

    if user_input:
        response = conversation.run(user_input)
        st.write("🧠 AI Tutor:", response)

    # Display conversation history
    st.subheader("Conversation History")
    st.write(st.session_state.memory.buffer)
