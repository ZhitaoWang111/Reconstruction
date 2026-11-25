"""Send one small reasoning request per model id through the DualCraft relay and print what
came back: which model answered, how long it took, and the token counts.

Runs on the DualCraft server from its repository (reads ``src/dualcraft/.env``)::

    cd /mnt/data/usr/wzt/submodules/dualcraft/dual-craft
    .venv/bin/python <this file> "七牛云/coding-openai/gpt-6-astra" gpt-6-astra
"""
from __future__ import annotations

import sys
import time

from dotenv import dotenv_values
from openai import OpenAI

PROMPT = ("A ladder has 7 rungs 30 cm apart and its bottom rung is 40 cm above the floor. "
          "How many cm above the floor is the top rung? Reply with the number only.")


def main() -> int:
    env = dotenv_values("src/dualcraft/.env")
    client = OpenAI(base_url=env["OPENAI_BASE_URL"], api_key=env["OPENAI_API_KEY"], timeout=180)
    for model in sys.argv[1:]:
        started = time.time()
        try:
            r = client.responses.create(model=model, input=PROMPT, reasoning={"effort": "high"},
                                        max_output_tokens=2000, store=False)
        except Exception as exc:
            print(f"{model}: FAILED {type(exc).__name__}: {str(exc)[:300]}")
            continue
        u = r.usage
        cached = getattr(u.input_tokens_details, "cached_tokens", None)
        reasoning = getattr(u.output_tokens_details, "reasoning_tokens", None)
        print(f"{model}: {time.time() - started:.1f}s status={r.status} answered_model={r.model!r} "
              f"text={r.output_text.strip()!r} input={u.input_tokens} cached={cached} "
              f"output={u.output_tokens} reasoning={reasoning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
