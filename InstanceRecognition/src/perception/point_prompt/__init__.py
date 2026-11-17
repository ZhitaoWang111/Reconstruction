"""Object perception by point prompts: a vision model clicks, SAM3 segments, the model keeps.

The alternative to the text-prompt perception (src.perception.label_frames + gpt_sam31,
GPT names categories and SAM3.1 segments each name). Here the model plans the objects in a
frame, clicks positive and negative points for each one, looks at SAM3's mask, and keeps
it, segments again, combines pieces of one unit, or skips it. It follows structural breaks
when deciding what one object is (a run of base cabinets with one worktop is one unit; a TV
cabinet beside a bookcase wall is two), which the text prompts could not do.

    clicks.py   SAM3 behind a lock, one frame's tools, the images the model sees (shared with
                src.structure.segment, which uses the same clicking for walls and floors)
    objects.py  one frame per conversation, free labels; the object runs
    source.py   a finished run -> an upstream mask source src.geometry.amb3r_mask_source lifts,
                and the renumbering the harness needs after lifting

Typical chain (scene0000_00, the run that fed harness-v15):

    $SAM31_PYTHON -m src.perception.point_prompt.objects --run full100_v2
    python -m src.perception.point_prompt.source pack --run full100_v2 \\
        --output artifacts/sources/scene0000_00/gpt_point_full100_v2
    python -m src.geometry.amb3r_mask_source --source artifacts/sources/scene0000_00/gpt_point_full100_v2 \\
        --output artifacts/sources/scene0000_00/scene_amb3r_point_full100_v2 --conf 0.3
    python -m src.perception.point_prompt.source renumber artifacts/sources/scene0000_00/scene_amb3r_point_full100_v2

Runs are written to artifacts/experiments/<run>/perception/point_prompt/.
"""
