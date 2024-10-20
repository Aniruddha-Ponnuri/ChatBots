from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from uuid import uuid4
import os
from openai import OpenAI
from groq import Groq
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware
from pydantic import Field

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add session middleware to FastAPI
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "your-secret-key"))

# Set up your API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

class ChatRequest(BaseModel):
    prompt: str
    model_choice: str = Field("GPT-4")

    class Config:
        protected_namespaces = ()  # Disable protected namespaces to resolve the warning

# Helper function to initialize or retrieve session memory
def get_memory(session_id: str, request: Request):
    if "memory" not in request.session:
        request.session["memory"] = {}
    if session_id not in request.session["memory"]:
        request.session["memory"][session_id] = ConversationBufferMemory()
    return request.session["memory"][session_id]

@app.post("/chat/")
async def chat(chat_request: ChatRequest, request: Request):
    session_id = request.cookies.get("session_id", str(uuid4()))
    request.session["session_id"] = session_id

    model_choice = chat_request.model_choice
    prompt = chat_request.prompt

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    max_tokens = 4000
    temperature = 0.02

    # Initialize the prompt template correctly
    chat_prompt = ChatPromptTemplate.from_template("{input}")
    
    # Retrieve or initialize session-specific memory buffer
    memory = get_memory(session_id, request)

    # Adding user message to memory for context tracking
    memory.chat_memory.add_user_message(prompt)
    previous_conversation_summary = memory.get_summary()

    # Prepare input for the LLM
    if previous_conversation_summary:
        prompt_with_context = f"Summary of previous conversation: {previous_conversation_summary}\n{prompt}"
    else:
        prompt_with_context = prompt

    # Run the LLM chain
    if model_choice == "GPT-4":
        llm_chain = LLMChain(llm=openai_client, prompt=chat_prompt, memory=memory)
    elif model_choice == "Groq LLama-3.1":
        llm_chain = LLMChain(llm=groq_client, prompt=chat_prompt, memory=memory)
    elif model_choice == "GPT-4o":
        openai_client.model = "gpt-4o"
        llm_chain = LLMChain(llm=openai_client, prompt=chat_prompt, memory=memory)
    else:
        raise HTTPException(status_code=400, detail="Invalid model choice")

    response = llm_chain.run(input={"input": prompt_with_context}, max_tokens=max_tokens, temperature=temperature)
    generated_text = response['choices'][0]['message']['content']

    # Adding the generated text to the memory
    memory.chat_memory.add_ai_message(generated_text)

    return {"generated_text": generated_text}
