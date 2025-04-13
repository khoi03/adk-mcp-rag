import os

import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.models.lite_llm import LiteLlm

from .tools import PromptLoader, MCPTools
from pydantic import BaseModel

# Load environment variables from .env file in the parent directory
# Place this near the top, before using env vars like API keys
load_dotenv('../.env')
# os.environ['OPENAI_API_KEY'] = "" # Set your OpenAI API key here

class Agents():
  """Manages agents"""

  def __init__(self):
    self.prompt_loaders = PromptLoader()
    self.prompt_configs = self.prompt_loaders._load_base_config()
    self.mcp_tools = MCPTools()

  # --- RAG Agent Definition ---
  async def get_rag_agent_async(self):
      """Creates an ADK Agent equipped with tools from the MCP Server."""
      tools, exit_stack = await self.mcp_tools.get_tools_async("http://localhost:8000/sse")
      print(f"Fetched {len(tools)} tools from MCP server.")
      print(tools)
      
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
  