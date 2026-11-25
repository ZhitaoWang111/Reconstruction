"""Step 3, the appearance agent.

    python -m structuregen.agent.run [run dir] [--fake] [--max-usd 20] [--resume <replay dir>]
    python -m structuregen.agent.replay <agent run> <new run> [--stop-before wrapup]
    python -m structuregen.agent.amend <agent run> --paint <seg> <face> <r,g,b> --reason "..."
    python -m structuregen.agent.compare <scene.blend> <frame> ... --out <dir>

run.py      prompts, tool schemas, the per-body loop (floors, walls, wrap-up), budget, fake replies
session.py  the harness: declare / execute_blender_code / compare / finish_segment / commit, order,
            neighbour hints; talks to blender/blender_worker.py over stdin/stdout
llm.py      the gateway client (gpt-6-astra), cost, logged calls
compare.py  composes the photo | model picture from the Blender passes
replay.py   re-applies a run's edits to a rebuilt white model, without the model
amend.py    hand edits after a run, recorded in amendments.json
blender/    runs inside Blender: blender_worker.py, sg_ops.py (sg.paint / strip / frame / tiles),
            checks.py, blender_compare.py, blender_amend.py
"""
