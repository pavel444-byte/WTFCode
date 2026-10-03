import streamlit as st
import os
from dotenv import load_dotenv

from main import CodeAssist, fetch_available_models, handle_context_command

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

    st.markdown("""
        <style>
        .stApp { background: #0b0d10; color: #e6edf3; }
        [data-testid="stSidebar"] { background: #11151a; border-right: 1px solid #27313a; }
        [data-testid="stChatMessage"] {
            border: 1px solid #27313a; border-radius: 10px; background: #11151a;
        }
        .wtf-header { font-family: ui-monospace, monospace; margin-bottom: 1rem; }
        .wtf-header strong { color: #7ee787; font-size: 1.6rem; }
        .wtf-header span { color: #8b949e; margin-left: .75rem; }
        .wtf-status { color: #7ee787; font-family: ui-monospace, monospace; }
        </style>
        <div class="wtf-header"><strong>WTFCode</strong><span>project-aware coding agent</span></div>
    """, unsafe_allow_html=True)
    
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

        modes = ["agent", "plan", "ask"]
        st.session_state.mode = st.selectbox(
            "Mode", modes, index=modes.index(st.session_state.mode)
        )
        
        st.divider()
        st.info(f"Provider: {st.session_state.assistant.provider}")
        st.info(f"Model: {st.session_state.assistant.model}")
        st.markdown(f'<div class="wtf-status">● {st.session_state.mode.upper()} · READY</div>', unsafe_allow_html=True)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.assistant.clear_context()
            st.rerun()

        st.caption("Use `/context clear` to reset AI context, or `/context image {list|add|remove}` to manage one-shot image attachments.")

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

        if prompt.strip().startswith("/context"):
            result = handle_context_command(st.session_state.assistant, prompt.strip())
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.mode == "agent":
                with st.spinner("Agent is thinking and acting..."):
                    content = st.session_state.assistant.run_agent(prompt, render=False)
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
            elif st.session_state.mode == "plan":
                with st.spinner("Plan Mode is exploring the project (read-only)..."):
                    content = st.session_state.assistant.plan(prompt, render=False)
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
            else:
                with st.spinner("Thinking..."):
                    content = st.session_state.assistant.ask_only(prompt, render=False)
                    st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})

if __name__ == "__main__":
    main()
