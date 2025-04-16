import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google.genai import types
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters

from pydantic import BaseModel

# Load environment variables from .env file in the parent directory
load_dotenv('.env')

# Global thread and loop for MCP tools
_mcp_thread = None
_mcp_loop = None
_mcp_tools_ready = threading.Event()
_mcp_tools_result = None
_mcp_tools_error = None

class MCPTools(BaseModel):
    """Manages tools from MCP Server with google adk"""

    def __init__(self):
        pass

    # --- Import Tools from MCP Server ---
    async def get_tools_async(self, sse_url: str):
        """Gets tools from MCP Server asynchronously."""
        print("Attempting to connect to MCP server...")
        tools, exit_stack = await MCPToolset.from_server(
            # Use SseServerParams for remote servers
            connection_params=SseServerParams(url=sse_url)
        )
        print("MCP Toolset created successfully.")
        # MCP requires maintaining a connection to the local MCP Server.
        # exit_stack manages the cleanup of this connection.
        return tools, exit_stack
    
    def _mcp_thread_main(self, sse_url):
        """Main function for the MCP thread."""
        global _mcp_loop, _mcp_tools_result, _mcp_tools_error
        
        try:
            # Create a new event loop for this thread
            _mcp_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_mcp_loop)
            
            # Run the async function and store the result
            _mcp_tools_result = _mcp_loop.run_until_complete(self.get_tools_async(sse_url))
            
            # Signal that tools are ready
            _mcp_tools_ready.set()
            
            # Keep the loop running to handle future requests
            _mcp_loop.run_forever()
        except Exception as e:
            # Store any error
            _mcp_tools_error = e
            _mcp_tools_ready.set()  # Signal even on error
            print(f"Error in MCP thread: {e}")
        finally:
            # Clean up loop on exit
            if _mcp_loop and _mcp_loop.is_running():
                _mcp_loop.stop()
            
            # Note: We intentionally do NOT close the loop here
            # because we want it to stay open for tool invocations
    
    def get_tools(self, sse_url: str):
        """Gets tools from MCP Server using a persistent background thread."""
        global _mcp_thread, _mcp_tools_result, _mcp_tools_error
        
        # Reset global state
        _mcp_tools_ready.clear()
        _mcp_tools_result = None
        _mcp_tools_error = None
        
        # Start MCP thread if not already running
        if _mcp_thread is None or not _mcp_thread.is_alive():
            print("Starting MCP tools thread...")
            _mcp_thread = threading.Thread(
                target=self._mcp_thread_main,
                args=(sse_url,),
                daemon=True  # Make thread daemon so it exits when main program exits
            )
            _mcp_thread.start()
        
        # Wait for tools to be ready
        print("Waiting for MCP tools to be ready...")
        _mcp_tools_ready.wait()
        
        # Check for errors
        if _mcp_tools_error:
            raise _mcp_tools_error
        
        # Return the result
        return _mcp_tools_result