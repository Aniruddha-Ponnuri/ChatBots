import os
import chainlit as cl
from openai import OpenAI
from groq import Groq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# Set up your API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

# Initialize memory buffer
memory = ConversationBufferMemory()

@cl.on_chat_start
async def on_chat_start():
    if 'model_choice' not in cl.session:
        cl.session['model_choice'] = 'GPT-4'

    options = ["GPT-4", "GPT-4o", "Groq LLama-3.1"]
    user_choice = await cl.ask_user(
        content="Choose a model to start with:",
        choices=options
    )
    cl.session['model_choice'] = user_choice['content']
    await cl.Message(content=f"You chose {user_choice['content']} model.").send()

@cl.on_message
async def main(prompt):
    if 'model_choice' not in cl.session:
        options = ["GPT-4", "GPT-4o", "Groq LLama-3.1"]
        user_choice = await cl.ask_user(
            content="Choose a model for this message:",
            choices=options
        )
        cl.session['model_choice'] = user_choice['content']

    model_choice = cl.session['model_choice']

    # Slider for max tokens
    max_tokens = 150  # Default value
    temperature = 0.02  # Default value

    if prompt:
        chat_prompt = ChatPromptTemplate.from_template("{{input}}")
        chat_prompt.input_variables["input"] = prompt

        # Add memory buffer content to prompt
        memory.add_user_message(prompt)
        previous_conversation_summary = memory.get_summary()

        # Update chat prompt with conversation summary
        if previous_conversation_summary:
            chat_prompt.input_variables["input"] = f"Summary of previous conversation: {previous_conversation_summary}\n{prompt}"

        if model_choice == "GPT-4":
            llm_chain = LLMChain(llm=openai_client, prompt=chat_prompt, memory=memory)
            response = llm_chain.run(max_tokens=max_tokens, temperature=temperature)
            generated_text = response['choices'][0]['message']['content']

        elif model_choice == "Groq LLama-3.1":
            llm_chain = LLMChain(llm=groq_client, prompt=chat_prompt, memory=memory)
            response = llm_chain.run(max_tokens=max_tokens, temperature=temperature)
            generated_text = response['choices'][0]['message']['content']

        elif model_choice == "GPT-4o":
            openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], model="gpt-4o")
            llm_chain = LLMChain(llm=openai_client, prompt=chat_prompt, memory=memory)
            response = llm_chain.run(max_tokens=max_tokens, temperature=temperature)
            generated_text = response['choices'][0]['message']['content']

        # Add assistant message to memory
        memory.add_assistant_message(generated_text)

        # Display the generated text
        await cl.Message(content="Generated Text:").send()
        await cl.Message(content=generated_text).send()