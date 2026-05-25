import streamlit as st
import os
from dotenv import load_dotenv

from main import CodeAssist, fetch_available_models

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assistant" not in st.session_state:
        load_dotenv()
        provider = os.getenv("PROVIDER", "openai")
        model = os.getenv("MODEL")
        st.session_state.assistant = CodeAssist(provider=provider, model=model)
    if "mode" not in st.session_state:
        st.session_state.mode = "agent"
    if "available_models" not in st.session_state:
        st.session_state.available_models = fetch_available_models(st.session_state.assistant.provider)

def main():
    st.set_page_config(page_title="WTFCode Web", page_icon="🤖", layout="wide")
    init_session_state()

    st.title("🤖 WTFCode Web")
    
    with st.sidebar:
        st.header("Settings")
        
        # Provider Selection
        current_provider = st.session_state.assistant.provider
        providers = ["openai", "anthropic", "openrouter", "gemini", "azure_openai", "llama"]
        new_provider = st.selectbox(
            "Provider", 
            providers, 
            index=providers.index(current_provider) if current_provider in providers else 0
        )
        
        if new_provider != current_provider:
            with st.spinner(f"Switching to {new_provider}..."):
                st.session_state.assistant = CodeAssist(provider=new_provider)
                st.session_state.available_models = fetch_available_models(new_provider)
            st.rerun()

        # Model Selection
        if st.session_state.available_models:
            current_model = st.session_state.assistant.model
            try:
                model_index = st.session_state.available_models.index(current_model)
            except ValueError:
                model_index = 0
                
            new_model = st.selectbox(
                "Model", 
                st.session_state.available_models, 
                index=model_index
            )
            
            if new_model != current_model:
                st.session_state.assistant.model = new_model
                st.success(f"Model changed to {new_model}")

        st.session_state.mode = st.selectbox("Mode", ["agent", "ask"], index=0 if st.session_state.mode == "agent" else 1)
        
        st.divider()
        st.info(f"Provider: {st.session_state.assistant.provider}")
        st.info(f"Model: {st.session_state.assistant.model}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.assistant.reset_history()
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        if prompt.strip() == "/exit":
            st.warning("Exiting WTFCode...")
            # Kill the process
            os._exit(0)
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.mode == "agent":
                with st.spinner("Agent is thinking and acting..."):
                    content = st.session_state.assistant.run_agent(prompt, render=False)
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
            else:
                with st.spinner("Thinking..."):
                    content = st.session_state.assistant.ask_only(prompt, render=False)
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})

if __name__ == "__main__":
    main()
