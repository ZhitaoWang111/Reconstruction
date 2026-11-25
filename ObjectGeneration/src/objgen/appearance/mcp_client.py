"""Local stdio MCP connection to the fixed third-party checkout."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def blender_session(repo, port):
    env = dict(os.environ, PYTHONPATH=str(Path(repo) / "src"), BLENDER_HOST="127.0.0.1",
               BLENDER_PORT=str(port), BLENDER_MCP_DISABLE_TELEMETRY="1")
    params = StdioServerParameters(command=sys.executable,
                                  args=["-m", "blender_mcp.server", "--port", str(port)], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def execute_template(session, code, intent):
    result = await session.call_tool("execute_blender_code", {"code": code, "user_prompt": intent})
    text = "\n".join(c.text for c in result.content if c.type == "text")
    if result.isError or not text.startswith("Code executed successfully:"):
        raise RuntimeError("Blender tool failed: " + text[:1500])
    return text
