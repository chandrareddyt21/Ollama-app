import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACKING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot With OLLAMA"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user", "question:{question}")
    ]
)


def generate_response(question, llm, temperature, max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer



# Title of the app
st.title("Enhanced Q&A Chatbot With OLLAMA")

# Sidebar for settings
# Drop down to select a various OPENAI models
llm = st.sidebar.selectbox("Select an OLLAMA model",["moondream","mistral"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Hey! I am ready to answer your question. Please ask any question...")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(
        question=user_input, 
        llm=llm, 
        temperature=temperature, 
        max_tokens= max_tokens
        )
    st.write(response)
else:
    st.write("Please provide a question to answer...")