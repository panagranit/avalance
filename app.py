# import packages
from dotenv import load_dotenv
import openai
import os
import streamlit as st


st.title("Hello Generative AI ")
st.write("This is a simple app to demonstrate OpenAI API usage.")
# load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "user", "content": "Explain generative AI in one sentence."}  # Prompt
    ],
    temperature=0.7,  # A bit of creativity
    max_output_tokens=100  # Limit response length
)

# print the response from OpenAI
st.write(response.output[0].content[0].text)

