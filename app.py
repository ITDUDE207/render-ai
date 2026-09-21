import os
import sys
import json
import subprocess
import streamlit as sns
import streamlit as st
from openai import OpenAI

# Initialize client using Groq cloud configurations
if not os.getenv("GROQ_API_KEY"):
    st.error("Error: GROQ_API_KEY environment variable not configured inside .env or Render settings.")
    st.stop()

client = OpenAI(
    base_url="https://groq.com",
    api_key=os.getenv("GROQ_API_KEY")
)

WORKSPACE_DIR = os.path.abspath("./workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
MODEL_NAME = "llama-3.3-70b-versatile" 

SYSTEM_PROMPT = """You are an autonomous AI software engineer. Your goal is to solve the user's task.
You have access to two capabilities:
1. Run a bash command in your workspace.
2. Write/overwrite a file in the workspace.

You MUST respond strictly in a raw JSON object format so your parsing engine can read it. 
Do not wrap your answer in markdown code blocks like ```json ... ```. Just return raw JSON.

If you need to run a command:
{
    "thought": "I need to check the current directory contents.",
    "action": "RunCommand",
    "command": "ls -la"
}

If you need to create or update a file:
{
    "thought": "I will create a basic app file.",
    "action": "WriteFile",
    "path": "app.py",
    "content": "print('Hello World')"
}

If you are completely finished with the task:
{
    "thought": "I have completed everything successfully.",
    "action": "Complete",
    "message": "The app has been built and tested successfully."
}
"""

def execute_sandbox_command(command):
    try:
        res = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30, 
            cwd=WORKSPACE_DIR
        )
        return f"Exit Code: {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."

def write_sandbox_file(rel_path, content):
    full_path = os.path.join(WORKSPACE_DIR, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return f"Successfully wrote file to {rel_path}"

def clean_and_parse_json(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return json.loads(text.strip())

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="OpenHands Lite Clone")
st.title("🤖 OpenHands Cloud Engine (Powered by Groq)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = []

left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("Chat Interface")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_prompt := st.chat_input("What software engineering task should I do?"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)
            
        # Agent execution flow
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in st.session_state.messages:
            api_messages.append({"role": m["role"], "content": m["content"]})
            
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            
            for iteration in range(10):
                status_placeholder.text(f"Thinking (Step {iteration+1}/10)...")
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages,
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    
                    response_text = response.choices.message.content
                    action_data = clean_and_parse_json(response_text)
                    
                    thought = action_data.get('thought')
                    action = action_data.get('action')
                    
                    st.session_state.logs.append(f"🧠 Thought: {thought}")
                    
                    if action == "Complete":
                        msg = action_data.get('message')
                        st.write(f"✅ **Task Complete!** {msg}")
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        break
                        
                    elif action == "RunCommand":
                        cmd = action_data.get("command")
                        st.session_state.logs.append(f"💻 Action: Running `{cmd}`")
                        tool_output = execute_sandbox_command(cmd)
                        
                    elif action == "WriteFile":
                        path = action_data.get("path")
                        content = action_data.get("content")
                        st.session_state.logs.append(f"📝 Action: Writing file `{path}`")
                        tool_output = write_sandbox_file(path, content)
                        
                    api_messages.append({"role": "assistant", "content": response_text})
                    api_messages.append({"role": "user", "content": f"Observation:\n{tool_output}"})
                    
                except Exception as e:
                    st.error(f"Error in execution loop: {e}")
                    break
            status_placeholder.empty()
            st.rerun()

with right_col:
    st.header("Workspace Action Logs")
    if st.button("Clear Log View"):
        st.session_state.logs = []
    for log in st.session_state.logs:
        st.caption(log)
