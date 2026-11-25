"""Run ``dualcraft.agent.run.cli`` unchanged, with two additions for relay route ids and
token bookkeeping.  DualCraft's repository is not modified.

1. Pricing for routed model ids.  DualCraft prices a model by the start of its id
   (``gpt-6-astra...``).  A relay route id such as ``七牛云/coding-openai/gpt-6-astra``
   matches nothing, and then DualCraft keeps no cost tracker: no ``cost.json``, no
   per-turn output tokens on disk, and ``--max-cost-usd`` never trips.  Here the price
   lookup also tries the part after the last ``/`` (``gpt-6-astra``), so the run is
   priced, capped and recorded the same way as with the bare id.  Nothing else in
   DualCraft depends on the model id's prefix (checked at 45fc1b10: only
   ``pricing_for_provider_model`` in ``harness.py``).

2. Object schemas without ``properties``.  Some relay routes (``七牛云/...``, and the
   default route on 2026-11-11) validate tool schemas strictly and refuse an object node
   that has no ``properties``: "Invalid schema for function 'blender_commit': At schema
   path /properties/joint_sign_overrides/type/0, object schema must define properties".
   DualCraft sends its free-key maps that way (``dict[str, int]``).  Every such node gets
   an empty ``properties`` (which still accepts every value the map accepts) as the tools
   are converted for the request; at 45fc1b10 that is ``blender_commit.joint_sign_overrides``
   alone.  This is the change ``third_party_patches/dual-craft-openai_codec-object-properties.patch``
   made in DualCraft's source, applied at run time instead.

3. Raw usage per model response.  DualCraft's cost tracker keeps input / cached / output
   tokens but drops reasoning tokens.  Every usage the OpenAI provider extracts is also
   appended as one JSON line to ``$DUALCRAFT_USAGE_LOG`` (when set), with a timestamp and
   the response's own model name.

Usage (from the DualCraft repository root, ``PYTHONPATH=src``)::

    DUALCRAFT_USAGE_LOG=<run>/usage.jsonl .venv/bin/python <this file> <dualcraft-run args...>
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time


def _route_suffix(model_id: str) -> str:
    return (model_id or "").rsplit("/", 1)[-1]


def _declare_object_properties(node):
    """Give every object-typed schema node without ``properties`` an empty one, recursively."""
    if isinstance(node, list):
        return [_declare_object_properties(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _declare_object_properties(value) for key, value in node.items()}
    kind = out.get("type")
    is_object = kind == "object" or (isinstance(kind, list) and "object" in kind)
    if is_object and not isinstance(out.get("properties"), dict):
        out["properties"] = {}
    return out


def install() -> None:
    from dualcraft.providers import cost, openai_codec
    from dualcraft.providers import openai as openai_provider

    original_convert = openai_codec.convert_tools

    def convert_tools(tools):
        converted = original_convert(tools)
        for tool in converted:
            if isinstance(tool, dict) and isinstance(tool.get("parameters"), dict):
                tool["parameters"] = _declare_object_properties(tool["parameters"])
        return converted

    openai_codec.convert_tools = convert_tools

    original_pricing = cost.pricing_for_provider_model

    def pricing_for_provider_model(provider: str, model_id: str):
        found = original_pricing(provider, model_id)
        if found is None and "/" in (model_id or ""):
            found = original_pricing(provider, _route_suffix(model_id))
        return found

    cost.pricing_for_provider_model = pricing_for_provider_model
    # harness.py imported the function by name; rebind that name too.
    from dualcraft.agent.loop import harness
    harness.pricing_for_provider_model = pricing_for_provider_model

    log_path = os.environ.get("DUALCRAFT_USAGE_LOG")
    if not log_path:
        return
    lock = threading.Lock()
    # Hook the provider method, not openai_codec.extract_usage: convert_response calls the
    # codec function again for the same response, which would log every response twice.
    original_extract = openai_provider.OpenAILLM._extract_usage

    def extract_usage(self, response):
        usage = original_extract(self, response)
        try:
            model = response.get("model") if isinstance(response, dict) else getattr(response, "model", None)
            status = response.get("status") if isinstance(response, dict) else getattr(response, "status", None)
            line = json.dumps({"ts": round(time.time(), 3), "model": model, "status": status,
                               "usage": usage}, ensure_ascii=False)
            with lock, open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # bookkeeping must never break a run
            pass
        return usage

    openai_provider.OpenAILLM._extract_usage = extract_usage


def main() -> int:
    install()
    from dualcraft.agent.run.cli import main as cli_main
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
