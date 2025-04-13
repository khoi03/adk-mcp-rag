import os

# ./adk_agent_samples/mcp_agent/agent.py
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters

from pydantic import BaseModel
# Load environment variables from .env file in the parent directory
# Place this near the top, before using env vars like API keys
load_dotenv('.env')

class MCPTools(BaseModel):
  """Mananges tools from MCP Server with google adk"""

  def __init__(self):
    pass

  # --- Import Tools from MCP Server ---
  async def get_tools_async(self, sse_url: str):
    """Gets tools from MCP Server."""
    print("Attempting to connect to MCP server...")
    tools, exit_stack = await MCPToolset.from_server(
        # Use SseServerParams for remote servers
        connection_params=SseServerParams(url=sse_url)
    )
    print("MCP Toolset created successfully.")
    # MCP requires maintaining a connection to the local MCP Server.
    # exit_stack manages the cleanup of this connection.
    return tools, exit_stack