"""Wall geometry from 2D structure masks: lift, rays, planes, closure.

Turns the kept masks of a structure segmentation run (labels wall / floor / door / window,
one result.json per frame) into the closed walls the wall handoff is written from. Four
steps, each a module with a ``main``, each reading the previous one's files:

    python -m src.structure.lift   --run struct_v4_win5    # masks -> 3D points with normals
    python -m src.structure.rays                           # every pixel of every frame -> rays
    python -m src.structure.walls  --run struct_v4_win5    # floor plane, then wall planes
    python -m src.structure.close  --run struct_v4_win5    # exterior/interior, outline, height, thickness
    python -m src.handoff.walls_from_harness --run struct_v4_win5

Intermediate files go to ``artifacts/structure/<run>/`` (rays, which do not depend on the
run, to ``artifacts/structure/``). No ScanNet annotation is read anywhere in the chain.
"""
