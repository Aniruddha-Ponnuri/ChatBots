import os
import streamlit as st
from openai import OpenAI
from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

# Set up your API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
groq_client = Groq()

# Streamlit app
st.title("Chat Bot")
st.write("Generate text using OpenAI's GPT-4 or Groq's LLama-3.1")

# Model selection
model_choice = st.selectbox("Choose a model", ["GPT-4", "GPT-4o","Groq LLama-3.1"])

# User input for the prompt
prompt = st.text_area("Enter your prompt here:")

# Slider for max tokens
max_tokens = 1000
#st.slider("Max tokens:", min_value=10, max_value=1000, value=150)

# Slider for temperature
temperature = 0.01
#st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.02)

if st.button("Generate"):
    if prompt:
        if model_choice == "GPT-4":
            # Call the GPT-4 API
            completion = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None
            )

            # Get the generated text
            generated_text = completion.choices[0].message.content

        elif model_choice == "Groq LLama-3.1":
            # Call the Groq LLama-3.1 API
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant"
            )

            # Get the generated text
            generated_text = response.choices[0].message.content

        elif model_choice == "GPT-4o":
            # Call the GPT-4o API
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None
            )

            # Get the generated text
            generated_text = completion.choices[0].message.content

        # Display the generated text
        print(completion)
        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.error("Please enter a prompt.")
