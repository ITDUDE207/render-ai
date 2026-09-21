import os
import sys
import json
import subprocess
import streamlit as st
from openai import OpenAI

# Initialize client using Groq cloud configurations
if not os.getenv("GROQ_API_KEY"):
    st.error("Error: GROQ_API_KEY environment variable not configured inside Render settings.")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
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
        return {
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "stdout": "", "stderr": "Error: Command timed out after 30 seconds."}

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

# --- Streamlit Layout Configuration ---
st.set_page_config(layout="wide", page_title="HyperDev Engine")

# App header themed like OpenHands workspace
st.markdown("### 🖥️ **HyperDev Workspace** <span style='color:#a0a0a0; font-size:14px;'>Powered by Groq</span>", unsafe_allow_html=True)
st.divider()

# Core State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []

# Split Workspace Columns (Left = Chat Console, Right = Sandbox Workspace Logs)
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("#### **User Console**")
    
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input controller
    if user_prompt := st.chat_input("Ask HyperDev to build something..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)
            
        # Build API compilation context window
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in st.session_state.messages:
            api_messages.append({"role": m["role"], "content": m["content"]})
            
        # Primary Execution Loop Container
        with st.chat_message("assistant"):
            agent_output_area = st.empty()
            loop_status = st.progress(0, text="Agent initialized...")
            
            # Step Loop limits continuous execution limits
            for iteration in range(8):
                step_num = iteration + 1
                loop_status.progress(int((step_num/8)*100), text=f"Executing Step {step_num}/8...")
                
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=api_messages,
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    
                    response_text = response.choices.message.content
                    action_data = clean_and_parse_json(response_text)
                    
                    thought = action_data.get('thought', 'Processing layout details...')
                    action = action_data.get('action', 'Complete')
                    
                    # Update active traces in real time
                    st.session_state.agent_trace.append({
                        "type": "Thought",
                        "title": f"Step {step_num}: Thought Process",
                        "body": thought
                    })
                    
                    if action == "Complete":
                        msg = action_data.get('message', 'Task completed.')
                        agent_output_area.markdown(f"### 🎉 **Task Complete**\n\n{msg}")
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        break
                        
                    elif action == "RunCommand":
                        cmd = action_data.get("command")
                        st.session_state.agent_trace.append({
                            "type": "Action",
                            "title": f"Step {step_num}: Executing Command",
                            "body": f"\$ {cmd}"
                        })
                        
                        # Process system operations
                        cmd_res = execute_sandbox_command(cmd)
                        tool_output = f"Exit Code: {cmd_res['exit_code']}\nSTDOUT:\n{cmd_res['stdout']}\nSTDERR:\n{cmd_res['stderr']}"
                        
                        st.session_state.agent_trace.append({
                            "type": "Terminal Output",
                            "title": f"Step {step_num}: Terminal Response",
                            "body": tool_output
                        })
                        
                    elif action == "WriteFile":
                        path = action_data.get("path")
                        content = action_data.get("content")
                        st.session_state.agent_trace.append({
                            "type": "Action",
                            "title": f"Step {step_num}: Writing Source Code",
                            "body": f"File target: `{path}`"
                        })
                        
                        tool_output = write_sandbox_file(path, content)
                        
                        st.session_state.agent_trace.append({
                            "type": "File System Output",
                            "title": f"Step {step_num}: Disk Summary",
                            "body": tool_output
                        })
                        
                    # Feed execution trace back into model state context
                    api_messages.append({"role": "assistant", "content": response_text})
                    api_messages.append({"role": "user", "content": f"Observation:\n{tool_output}"})
                    
                except Exception as e:
                    st.error(f"Execution Error encountered: {e}")
                    break
            
            loop_status.empty()
            st.rerun()

with right_col:
    st.markdown("#### 🛠️ **Execution Trace & Sandbox Terminal**")
    
    if st.button("Reset Terminal Views", use_container_width=True):
        st.session_state.agent_trace = []
        st.rerun()
        
    st.divider()
    
    # Render historical structural actions into diagnostic widgets
    if not st.session_state.agent_trace:
        st.info("No active operations executed yet. Issue a request in the console to trigger steps.")
    else:
        for trace in reversed(st.session_state.agent_trace):
            if trace["type"] == "Thought":
                with st.expander(f"🧠 {trace['title']}", expanded=True):
                    st.write(trace["body"])
            elif trace["type"] == "Action":
                with st.expander(f"⚙️ {trace['title']}", expanded=True):
                    st.code(trace["body"], language="bash")
            else:
                with st.expander(f"📟 {trace['title']}", expanded=False):
                    st.code(trace["body"])
