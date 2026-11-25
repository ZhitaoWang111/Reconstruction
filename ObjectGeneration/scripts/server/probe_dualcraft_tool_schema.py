"""Ask the DualCraft relay whether it accepts DualCraft's own blender_commit tool schema.

Runs on the DualCraft server, inside its repository (``PYTHONPATH=src``)::

    cd /mnt/data/usr/wzt/submodules/dualcraft/dual-craft
    PYTHONPATH=src .venv/bin/python <this file>

``blender_commit`` has an optional free-key map (``joint_sign_overrides``) that DualCraft's
codec sends as ``type: [object, null]`` with ``additionalProperties`` and no
``properties``.  On 2026-11-11 the relay refused that at turn 1 ("object schema must
define properties").  This sends one minimal request carrying only that tool, converted
by the checked-out codec exactly as a run would, and prints whether the relay accepted it.
It costs one short reply.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import sys

from dualcraft.agent.tools import build_tool_registry
from dualcraft.config import default_model_from_env, load_repo_env
from dualcraft.providers.openai import OpenAILLM


def main() -> int:
    # The repository's .env (Blender, display) and the relay's src/dualcraft/.env (key,
    # base URL, DUALCRAFT_MODEL), as a run reads them.
    load_repo_env(Path.cwd())
    load_repo_env(Path.cwd() / "src" / "dualcraft")
    registry = build_tool_registry("openai", sdk_package="sdk", blender_session=object())
    llm = OpenAILLM(model_id=default_model_from_env())
    tools = [t for t in llm._convert_tools(registry.get_tool_schemas()) if t["name"] == "blender_commit"]
    if not tools:
        print("blender_commit not registered")
        return 2
    node = tools[0]["parameters"]["properties"]["joint_sign_overrides"]
    print("schema sent for joint_sign_overrides:", json.dumps(node)[:200])
    try:
        request = llm._client.responses.create(
            model=llm.model_id, input="Reply with the single word OK. Do not call any tool.",
            tools=tools, tool_choice="none", max_output_tokens=200, reasoning={"effort": "low"},
            store=False)
        # DualCraft builds an async client; nothing is sent until the coroutine is awaited.
        response = asyncio.run(request) if inspect.isawaitable(request) else request
    except Exception as exc:  # the relay's refusal comes back as an API error
        print("REFUSED:", type(exc).__name__, str(exc)[:400])
        return 1
    print("ACCEPTED: model", llm.model_id, "| status", response.status,
          "| text", (response.output_text or "").strip()[:40], "| usage", response.usage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
