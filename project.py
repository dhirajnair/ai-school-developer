import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langsmith import traceable
from langchain_community.tools.shell.tool import ShellTool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import subprocess
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')

ROOT_DIR = os.getcwd()


@tool
def check_format(file_path: str) -> str:
    """Checks the format of the file at the given path."""
    return subprocess.run(["prettier", "--check", file_path], check=True)

@tool
def create_readme(content: str) -> str:
    """Creates a new README.md file in the root directory."""
    return create_file("README.md", content, file_type="md")


@tool
def create_unittest(content: str) -> str:
    """Creates a new unittest file in the root directory."""
    return create_file("unittest.py", content, file_type="py")

@tool
def execute_unittest() -> str:
    """Executes the unittest file in the root directory."""
    return subprocess.run(["python", "unittest.py"], check=True)

@tool
def create_react_app_with_vite() -> str:
    """Creates a new React application using Vite in the 'app' directory."""
    try:
        subprocess.run(["npm", "create", "vite@latest", ".", "--template", "react"], check=True)

        return "React application created successfully."
    except subprocess.CalledProcessError as e:
        return f"Failed to create React application: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred creating React application: {str(e)}"

@tool
def create_directory(directory: str) -> str:
    """Creates a new writable directory with the given name if it does not exist."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chmod(directory, 0o700)  # Set permissions to read, write & execute by owner
            return f"Directory {directory} created successfully."
        else:
            subprocess.run(["chmod", "u+w", directory], check=True)
            return f"Directory {directory} already exists."
        
    except subprocess.CalledProcessError as e:
        return f"Failed to create or access the directory {directory}: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred for {directory}: {str(e)}"

@tool
def find_file(filename: str, path: str) -> Optional[str]:
    """Recursively searches for a file in the given path."""
    for root, dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root, filename)
    return None

@tool
def create_file(filename: str, content: str = "", directory=ROOT_DIR, file_type: str = "") -> str:
    """Creates a new file with specified file type and content in the specified directory."""
    valid_file_types = ['txt', 'md', 'json', 'py', 'js', 'jsx', 'ts', 'tsx', 'html', 'css', 'java', 'cpp', 'json', 'md']
    file_extension = os.path.splitext(filename)[1]
    if not file_extension:  # Check if filename does not have an extension
        if file_type in valid_file_types:
            filename += f".{file_type}"
        else:
            return f"Invalid file type. Supported types are: {', '.join(valid_file_types)}"
    if file_type in valid_file_types:
        full_path = os.path.join(ROOT_DIR, directory, filename)
        if os.path.exists(full_path):
            return f"File {filename} already exists in {directory}."
        with open(full_path, 'w') as file:
            file.write(content)
        return f"File {filename} created successfully in {directory}."
    else:
        return f"Invalid file type. Supported types are: {', '.join(valid_file_types)}"

@tool
def update_file(filename: str, content: str, directory: str = "") -> str:
    """Updates, appends, or modifies an existing file with new content."""
    full_path = os.path.join(ROOT_DIR, directory, filename)
    if os.path.exists(full_path):
        with open(full_path, 'a') as file:
            file.write(content)
        return f"File {filename} updated successfully in {directory}."
    else:
        return f"File {filename} does not exist in {directory}."

# List of tools to use
tools = [
    ShellTool(ask_human_input=True), 
    create_directory, 
    create_react_app_with_vite, 
    find_file, 
    create_file, 
    update_file,
    create_readme,
    create_unittest,
    execute_unittest,
    check_format
    # Add more tools if needed
]

# Configure the language model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert web developer.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)

# Create the agent
agent = (
    # Fill in the code to create the agent here
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main loop to prompt the user
while True:
    user_prompt = input("Prompt: ")
    list(agent_executor.stream({"input": user_prompt}))
