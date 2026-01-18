import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

# Add current directory to path so we can import from main
sys.path.append(str(Path(__file__).parent))

from main import CodeAssist, read_file, write_file, edit_file, execute_command, glob_search, git_commit, fetch_available_models

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
        new_provider = st.selectbox(
            "Provider", 
            ["openai", "anthropic", "openrouter", "gemini"], 
            index=["openai", "anthropic", "openrouter", "gemini"].index(current_provider)
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
            st.session_state.assistant.history = st.session_state.assistant.history[:1]
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
            response_placeholder = st.empty()
            
            # We need to capture the output of the agent which usually goes to console
            # For now, we'll adapt the run_agent/ask_only logic or just use the assistant's methods
            # Since run_agent prints to console, we might need a web-friendly version or capture stdout
            
            if st.session_state.mode == "agent":
                # Note: run_agent in main.py uses console.print. 
                # In a real app, we'd refactor CodeAssist to return values or use a callback.
                # For this implementation, we'll simulate the interaction.
                with st.spinner("Agent is thinking and acting..."):
                    # This is a simplified version for the web UI
                    # In a full implementation, we'd want to show tool calls in the UI
                    st.session_state.assistant.run_agent(prompt)
                    
                    # Get the last message from history
                    last_msg = st.session_state.assistant.history[-1]
                    content = ""
                    if hasattr(last_msg, 'content'):
                        if isinstance(last_msg.content, list):
                            content = last_msg.content[0].text
                        else:
                            content = last_msg.content or ""
                    
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
            else:
                with st.spinner("Thinking..."):
                    # ask_only also prints to console, let's extract the logic
                    assistant = st.session_state.assistant
                    messages = [
                        {"role": "system", "content": "You are a helpful coding assistant. Answer the question directly."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    content = ""
                    if assistant.provider in ["openai", "openrouter"]:
                        response = assistant.client.chat.completions.create(
                            model=assistant.model,
                            messages=messages
                        )
                        content = response.choices[0].message.content or ""
                    elif assistant.provider == "anthropic":
                        response = assistant.client.messages.create(
                            model=assistant.model,
                            max_tokens=4096,
                            messages=messages
                        )
                        content = response.content[0].text if response.content else ""
                    elif assistant.provider == "gemini":
                        response = assistant.client.generate_content(prompt)
                        content = response.text if hasattr(response, 'text') else ""
                    
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})

if __name__ == "__main__":
    main()
