import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Set up API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
groq_client = Groq()  # Correctly initialize Groq client

# Initialize memory
memory = ConversationBufferMemory()

# Streamlit app title and description
st.title("Conversational AI Chatbot")
st.subheader("Chat with an AI model of your choice")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Custom CSS for chat bubbles
st.markdown(
    """
    <style>
    .user-bubble {
        background-color: #007bff;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .bot-bubble {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
        color: white;
        float: left;
        clear: both;
    }
    .clearfix {
        overflow: auto;
    }
    .fixed-bottom-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        padding: 10px 0;
    }
    .input-area {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        background-color: #0e1117;
        border-top: 1px solid #444;
    }
    .input-area select, .input-area button {
        margin-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to summarize messages using Groq's LLAMA model with retry mechanism
def summarize_message_with_llama(message, retries=3, wait_time=5):
    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": f"Summarize the following message: {message}"}],
                model="llama-3.1-8b-instant"
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except groq.InternalServerError as e:
            if attempt < retries - 1:
                time.sleep(wait_time)  # Wait before retrying
            else:
                raise e

# Placeholder for the input field with the fixed class
input_container = st.empty()

def submit():
    # Define max_tokens and temperature
    max_tokens = 4000
    temperature = 0.02

    # Get user input
    user_input = st.session_state.user_input.strip()
    
    # Check if user input is empty
    if not user_input:
        st.session_state.history.append({"role": "bot", "content": "However, I don't see a message provided. Please provide the message you'd like me to summarize and include sample code for. I'll be happy to assist you.", "model": "System"})
        return
    
    # Display user input immediately
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Re-render the chat container with the updated history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="clearfix"><div class="user-bubble">User: {msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="clearfix"><div class="bot-bubble">Model: {msg["model"]}<br>{msg["content"]}</div></div>', unsafe_allow_html=True)

    # Summarize user input using Groq's LLAMA model
    try:
        user_message_summary = summarize_message_with_llama(user_input)
        st.session_state.history[-1]['content'] = user_message_summary  # Update the last user message with its summary
    except groq.InternalServerError:
        st.session_state.history.append({"role": "bot", "content": "The Groq API service is currently unavailable. Please try again later.", "model": "System"})
        return

    # Retrieve the previous conversation history
    previous_conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.history])
    
    # Construct the input prompt
    input_prompt = f"{previous_conversation}\nYou: {user_message_summary}"
    
    # Initialize the appropriate model
    model_choice = st.session_state.get('model_choice', 'Groq LLama-3.1')  # Default to LLAMA
    
    if model_choice == "GPT-4":
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": input_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None
        )
        generated_text = completion.choices[0].message.content

    elif model_choice == "Groq LLama-3.1":
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": input_prompt}],
            model="llama-3.1-8b-instant"
        )
        generated_text = response.choices[0].message.content

    elif model_choice == "GPT-4o":
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": input_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None
        )
        generated_text = completion.choices[0].message.content

    # Summarize bot response using Groq's LLAMA model and append it to session state history with the model choice
    bot_message_summary = summarize_message_with_llama(generated_text)
    st.session_state.history.append({"role": "bot", "content": bot_message_summary, "model": model_choice})

    # Clear the input field
    st.session_state.user_input = ""

# Render the input field in the fixed position
with input_container.container():
    # Input text box with enter key triggering submit
    st.text_input("Type your message here and press Enter:", key="user_input", placeholder="Type your message here...", on_change=submit)

    # Display model selector and submit button side by side
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    model_choice = st.selectbox("Choose a model:", ["Groq LLama-3.1", "GPT-4", "GPT-4o"], key="model_choice")
    st.button("Send", on_click=submit)
    st.markdown('</div>', unsafe_allow_html=True)
