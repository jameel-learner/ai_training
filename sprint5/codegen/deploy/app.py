import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with OpenRouter API
client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

# Streamlit app title
st.title("Simple Chatbot App")

# User input
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        # Prepare the message for the API
        messages = [{"role": "user", "content": user_input}]
        
        # Call the OpenRouter API for chat completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Correct model ID without provider prefix
            messages=messages
        )
        
        # Display the chatbot's response
        bot_response = response.choices[0].message.content
        st.text_area("Chatbot:", bot_response, height=200)
    else:
        st.warning("Please enter a message.")
