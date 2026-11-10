"""Streamlit UI for the ChatGPT-clone sample app."""

import os

import streamlit as st

import config
from chat import get_client, stream_response

st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON)
st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")

api_key = os.getenv("OPENROUTER_API_KEY") or st.sidebar.text_input("OpenRouter API key", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything"):
    if not api_key:
        st.warning("Add your OpenRouter API key in the sidebar.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        reply = st.write_stream(stream_response(get_client(api_key), st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": reply})
