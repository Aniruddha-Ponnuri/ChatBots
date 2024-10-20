import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from langchain.globals import set_verbose
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

# Set verbosity if needed
set_verbose(True)  # Enable verbose logging, adjust as needed

# Load environment variables
load_dotenv()

# Set up API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
groq_client = Groq()  # Correctly initialize Groq client

# Wrap OpenAI client in the LangChain LLM wrapper
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Initialize summary buffer memory with the wrapped LLM
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)  # Example with a token limit

# Streamlit app title and description
st.title("Conversational AI Chatbot")
st.subheader("Chat with an AI model of your choice")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "GPT-4"  # Default model

# Custom CSS for the footer with input field, send button, and model selector
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
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 10px 0;
        background-color: #1e1e1e;
        z-index: 1000;
        border-top: none;  /* Remove any top border */
        margin-top: 0;  /* Ensure no margin above */
    }
    .footer .stTextInput, .footer .stButton, .footer .stSelectbox {
        margin-bottom: 0;
        padding: 10px;
        background-color: #333;
        border: none;  /* Remove border from elements */
        color: white;
        border-radius: 10px;
    }
    .footer .stTextInput {
        width: 60%;
        margin-right: 10px;
    }
    .footer .stSelectbox {
        width: 20%;
        margin-right: 10px;
    }
    .footer .stButton {
        width: 100%;  /* Full width of the column */
        height: 50px;  /* Increase button height */
        font-size: 18px;  /* Increase font size */
        margin-top: 10px;  /* Adjust the vertical position */
    }
    </style>
    """,
    unsafe_allow_html=True
)

def submit():
    # Define max_tokens and temperature
    max_tokens = 4000
    temperature = 0.02

    try:
        # Get the user's input
        user_input = st.session_state.user_input

        if user_input:  # Ensure there is something to submit
            # Append user input to session state history
            st.session_state.history.append({"role": "user", "content": user_input})

            # Retrieve the summarized conversation history
            summarized_history = memory.buffer

            # Construct the input prompt
            input_prompt = f"{summarized_history}\nYou: {user_input}"
            
            # Initialize the appropriate model
            model_choice = st.session_state.model_choice
            
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

            # Append bot response with model name to session state history
            st.session_state.history.append({"role": "bot", "content": f"({model_choice}): {generated_text}"})

            # Update memory with the user input and bot's response
            memory.save_context({"input": user_input}, {"output": generated_text})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    finally:
        # Clear the input field
        st.session_state.user_input = ""

# Display the conversation history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-bubble">User: {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">Bot: {msg["content"]}</div>', unsafe_allow_html=True)

# Footer for input, model selector, and send button
st.markdown('<div class="footer">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([5, 2, 1])
with col1:
    user_input = st.text_input("Type a message", key="user_input", placeholder="Type your message here...", on_change=None)  # No on_change here
with col2:
    st.session_state.model_choice = st.selectbox("Choose a model:", ["GPT-4", "Groq LLama-3.1", "GPT-4o"], index=["GPT-4", "Groq LLama-3.1", "GPT-4o"].index(st.session_state.model_choice))
with col3:
    st.text("")
    st.text("")
    st.button("Send", on_click=submit)

st.markdown('</div>', unsafe_allow_html=True)
