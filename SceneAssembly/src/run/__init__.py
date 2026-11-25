"""M8 orchestration: one run = all batches in order, resumable, then a final physics pass and the export.

    .venv/bin/python src/run/main.py --name assembly-20261123        # start, or resume a stopped run
    .venv/bin/python src/run/main.py --name ... --batches 3            # only some batches (testing)
    .venv/bin/python src/run/main.py --name ... --export-only          # write scene.usda from the state

Decided 2026-11-23: resume restarts the interrupted batch from the previous checkpoint (a batch's
conversation cannot be continued halfway); the final whole-scene physics run reports and marks
failed what falls or drops, without undoing (no agent is left to fix it); a cost cap stops the run
before a batch would start past it; the export references structure.usdc and every object's
physics.usdc, failed objects included and marked.
"""
