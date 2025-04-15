import os

import asyncio
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode

from agents import Agents

agents = Agents()
# --- Step 3: Main Execution Logic ---
async def async_main():
  session_service = InMemorySessionService()
  # Artifact service might not be needed for this example
  artifacts_service = InMemoryArtifactService()

  session = session_service.create_session(
      state={}, app_name='mcp_filesystem_app', user_id='user_fs'
  )

  # TODO: Change the query to be relevant to YOUR specified folder.
  # e.g., "list files in the 'documents' subfolder" or "read the file 'notes.txt'"
  query = "Use the tool and tell me who Tran Dinh Khoi is?"
  print(f"User Query: '{query}'")
  content = types.Content(role='user', parts=[types.Part(text=query)])

  root_agent, exit_stack = await agents.get_rag_agent_async()

  runner = Runner(
      app_name='mcp_filesystem_app',
      agent=root_agent,
      artifact_service=artifacts_service, # Optional
      session_service=session_service,
  )
  stream_mode = StreamingMode.SSE
  print("Running agent...")
  events_async = runner.run_async(
      session_id=session.id, 
      user_id=session.user_id, 
      new_message=content,
      run_config=RunConfig(streaming_mode=stream_mode),
  )
  
  async for event in events_async:
    if event.is_final_response():
      print("\nFinal response received. Exiting loop.")
      break

    if event.content and event.content.parts:
        if event.get_function_calls():
            print("CALLING TOOL:", event.get_function_calls()[0].name)
        elif event.get_function_responses():
            print("GET TOOL RESPONSE SUCCESSFULLY")
            print(event.get_function_responses())
        elif event.content.parts[0].text:
          print(event.content.parts[0].text, flush=True, end="")

  # Crucial Cleanup: Ensure the MCP server process connection is closed.
  print("Closing MCP server connection...")
  await exit_stack.aclose()
  print("Cleanup complete.")

if __name__ == '__main__':
  try:
    asyncio.run(async_main())
  except Exception as e:
    print(f"An error occurred: {e}")