"""2D structure segmentation: a vision model clicks, SAM segments, the model keeps or retries.

The model is shown a short run of consecutive frames and plans the room's fabric in them -
`wall`, `floor`, `door`, `window` and nothing else. For each planned item it clicks
positive and negative points, SAM turns them into a mask, the model looks at the result and
keeps it, segments again, or skips it. Every kept mask belongs to one frame.

    python -m src.structure.segment.window --run struct_v4_win5 --frames-every 2   # SAM environment

Output, one directory per run under artifacts/structure/<run>/segmentation/:

    frame_<id>/result.json, masks/*.png, kept.jpg   what src.structure.lift reads
    window_<k>_<first>-<last>/                     the conversation: calls/, images/, window.json
    config.json, prompt.txt, summary.json

The prompt is versioned in src/structure/prompts/; the run records its hash.
The clicking itself (SAM3, the per-frame tools) is shared with the object perception and
lives in src.perception.point_prompt.clicks.
"""
