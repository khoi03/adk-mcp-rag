import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService  # Optional
from google.adk.models.lite_llm import LiteLlm

from .tools import PromptLoader, MCPTools
from pydantic import BaseModel

# Load environment variables from .env file in the parent directory
load_dotenv('./docker/.env', override=True)

# Global variable to store agent and exit_stack
global_root_agent = None
global_exit_stack = None

class Agents():
    """Manages agents"""

    def __init__(self):
        self.prompt_loaders = PromptLoader()
        self.prompt_configs = self.prompt_loaders._load_base_config()
        self.mcp_tools = MCPTools()

    async def get_tool_async(self):
        """Creates an ADK Agent equipped with tools from the MCP Server."""
        tools, exit_stack = await self.mcp_tools.get_tools_async(os.getenv('QRANT_MCP_SSE'))
        return tools, exit_stack

    def get_tool(self):
        """Synchronous version of get_tool_async."""
        return self.mcp_tools.get_tools(os.getenv('QRANT_MCP_SSE'))
    
    # --- RAG Agent Definition ---
    async def get_rag_agent_async(self):
        """Creates an ADK Agent equipped with tools from the MCP Server asynchronously."""
        tools, exit_stack = await self.mcp_tools.get_tools_async(os.getenv('QRANT_MCP_SSE'))
        print(f"Fetched {len(tools)} tools from MCP server.")
        
        root_agent = LlmAgent(
            model=LiteLlm(
                model='gpt-4o-mini', 
            ),
            name='ask_rag_agent',
            instruction=self.prompt_configs['ask_rag_agent']['instruction_prompt'],
            tools=[
                tools[1]
            ],
            generate_content_config=types.GenerateContentConfig(
                temperature=0.2,
            )
        )
        return root_agent, exit_stack
    
    def get_rag_agent(self):
        """Creates an ADK Agent equipped with tools from the MCP Server synchronously."""
        global global_root_agent, global_exit_stack
        
        # If agent already initialized, return it
        if global_root_agent is not None and global_exit_stack is not None:
            return global_root_agent, global_exit_stack
            
        # Use the persistent thread approach to get tools
        tools, exit_stack = self.mcp_tools.get_tools(os.getenv('QRANT_MCP_SSE'))
        print(f"Fetched {len(tools)} tools from MCP server.")
        
        # Create the agent
        root_agent = LlmAgent(
            model=LiteLlm(
                model='gpt-4o-mini', 
            ),
            name='ask_rag_agent',
            instruction=self.prompt_configs['ask_rag_agent']['instruction_prompt'],
            tools=[
                tools[1]
            ],
            generate_content_config=types.GenerateContentConfig(
                temperature=0.2,
            )
        )
        
        # Store in global variables for reuse
        global_root_agent = root_agent
        global_exit_stack = exit_stack
        
        return root_agent, exit_stack

# Initialize the agent using the persistent thread approach
try:
    # Try to get the agent synchronously
    agents = Agents()
    root_agent, exit_stack = agents.get_rag_agent()
except Exception as e:
    # Log the error but don't crash
    print(f"Error initializing agent: {e}")
    # Set empty values to prevent further errors
    root_agent = None
    exit_stack = None