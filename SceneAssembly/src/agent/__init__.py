"""M7 agent: the tools, the instructions, the opening message, and the conversation loop.

    from agent import run_batch
    summary = run_batch(index, ctx, out_dir)

Model: 七牛云/coding-openai/gpt-6-astra over the vibeg gateway (OpenAI Responses API),
called the same way as ObjectGeneration's wall agent. One conversation per batch (M6).
Tool calls are strict; the history is only ever appended to; images are not trimmed yet
(user decision 2026-11-23: no image-cost optimisation for now).
"""
from .loop import run_batch, Context

__all__ = ['run_batch', 'Context']
