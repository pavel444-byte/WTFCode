import os
import sys
import threading
import queue
from typing import Optional
import customtkinter as ctk
from PIL import Image
from main import CodeAssist, theme_manager, config
import io

class WTFCodeDesktop(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("WTFCode Desktop")
        self.geometry("1000x700")
        
        # Initialize AI Assistant
        provider = os.getenv("PROVIDER", config.get("provider", "openai"))
        model = os.getenv("MODEL", config.get("model", "gpt-4o"))
        self.assistant = CodeAssist(provider=provider, model=model)
        
        self.setup_ui()
        self.msg_queue = queue.Queue()
        self.after(100, self.process_queue)

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="WTFCode", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.mode_var = ctk.StringVar(value="agent")
        self.mode_menu = ctk.CTkOptionMenu(self.sidebar, values=["agent", "ask"], variable=self.mode_var)
        self.mode_menu.grid(row=1, column=0, padx=20, pady=10)

        self.provider_label = ctk.CTkLabel(self.sidebar, text="Provider:")
        self.provider_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.provider_var = ctk.StringVar(value=os.getenv("PROVIDER", config.get("provider", "openai")))
        self.provider_menu = ctk.CTkOptionMenu(self.sidebar, values=["openai", "anthropic", "google", "ollama"], variable=self.provider_var, command=self.update_assistant)
        self.provider_menu.grid(row=3, column=0, padx=20, pady=10)

        self.model_label = ctk.CTkLabel(self.sidebar, text="Model:")
        self.model_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        self.model_entry = ctk.CTkEntry(self.sidebar)
        self.model_entry.insert(0, os.getenv("MODEL", config.get("model", "gpt-4o")))
        self.model_entry.grid(row=5, column=0, padx=20, pady=10)
        self.model_entry.bind("<Return>", lambda e: self.update_assistant())

        self.update_button = ctk.CTkButton(self.sidebar, text="Update Settings", command=self.update_assistant)
        self.update_button.grid(row=6, column=0, padx=20, pady=20)

        # Chat Area
        self.chat_frame = ctk.CTkFrame(self)
        self.chat_frame.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        self.chat_display = ctk.CTkTextbox(self.chat_frame, state="disabled", wrap="word")
        self.chat_display.configure(font=ctk.CTkFont(family="Consolas", size=12))
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure tags for better formatting
        self.chat_display.tag_config("user", foreground="#1f538d", spacing1=10, spacing3=5)
        self.chat_display.tag_config("assistant", foreground="#dce4ee", spacing1=5, spacing3=10)
        self.chat_display.tag_config("system", foreground="#999999")
        self.chat_display.tag_config("separator", foreground="#444444")

        # Input Area
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=1, padx=20, pady=20, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your request here...")
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_entry.bind("<Return>", lambda e: self.send_message())

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", command=self.send_message, width=100)
        self.send_button.grid(row=0, column=1)

    def update_assistant(self, *args):
        provider = self.provider_var.get()
        model = self.model_entry.get().strip()
        self.assistant = CodeAssist(provider=provider, model=model)
        self.append_text(f"*** System: Switched to {provider} ({model}) ***")

    def append_text(self, text: str, tag: str = ""):
        self.chat_display.configure(state="normal")
        
        if text.startswith("You:"):
            self.chat_display.insert("end", "\n" + "─"*60 + "\n", "separator")
            self.chat_display.insert("end", text + "\n", "user")
        elif text.startswith("*** System:"):
            self.chat_display.insert("end", "\n" + text + "\n", "system")
        else:
            # If it's not a user message or system message, it's from the assistant
            # We strip the leading/trailing whitespace that might come from the capture
            clean_text = text.strip()
            if clean_text:
                self.chat_display.insert("end", clean_text + "\n", "assistant")
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def send_message(self):
        query = self.input_entry.get().strip()
        if not query:
            return
        
        self.input_entry.delete(0, "end")
        self.append_text(f"You: {query}")
        
        # Run AI in a separate thread to keep UI responsive
        threading.Thread(target=self.run_ai, args=(query,), daemon=True).start()

    def run_ai(self, query: str):
        mode = self.mode_var.get()
        
        # Capture stdout to get the rich console output
        output_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            if mode == "agent":
                self.assistant.run_agent(query)
            else:
                self.assistant.ask_only(query)
            
            sys.stdout = original_stdout
            full_output = output_capture.getvalue()
            
            # Clean up ANSI escape codes if any (though rich usually handles it)
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_output = ansi_escape.sub('', full_output)
            
            self.msg_queue.put(clean_output)
        except Exception as e:
            sys.stdout = original_stdout
            self.msg_queue.put(f"Error: {str(e)}")

    def process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.append_text(msg)
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = WTFCodeDesktop()
    app.mainloop()
