"""Structure generation: walls and floors with appearance, from the upstream walls handoff.

One USD per room for scene assembly: every wall segment an independent body, every room's floor
an independent body, all static colliders. Geometry is scripted; appearance is chosen by an agent
that looks at the handoff's key frames. Four steps, one subpackage each:

    whitebox/   1 white model: wall segments cut at butt joints, openings left open, room floors
                from the walls' outline, key-frame cameras. plan.json + walls.blend + USD.
    evidence/   2 what the agent is given: task card, numbered plan view, each wall's and each
                floor's key frames (original + outlined).
    agent/      3 appearance agent: floors, then walls, then the hidden faces; every edit checked
                in a headless Blender and answered with a photo | model compare picture.
                agent/blender/ holds the scripts that run inside that Blender (the sg.* operators,
                the checks, the compare passes).
    export/     4 the structure handoff: structure.usdc (/Structure/Walls, /Structure/Floors),
                textures/, structure.json, manifest.json, README.md.
    common/     handoff reader, wall-plane projection, the USD layout shared by the steps.
    config.py   paths (Blender, output, handoff root).

Launchers in scripts/: run_structure_build.sh (1+2), run_structure_agent.sh (3),
run_structure_export.sh (4). Files named blender_*.py run inside Blender, not as modules.
"""
