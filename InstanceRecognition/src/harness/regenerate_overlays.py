"""Re-render archived masks without altering original API evidence or decisions.

Run from InstanceRecognition: python -m src.harness.regenerate_overlays OUT [OUT ...]
"""
import html
import os
import colorsys
import json
import shutil
import sys
from pathlib import Path
from .observations import Observations, digest
from .state import save


def regenerate(out):
    out = Path(out).resolve()
    items = json.loads((out/'observations.json').read_text())
    data = Observations.__new__(Observations)
    data.source = out
    data.out = out/'numbered_overlay_preview'
    data.out.mkdir(exist_ok=True)
    data.views = {}
    data.items = items
    data.frames = {}
    by_frame = {}
    for key, item in items.items():
        by_frame.setdefault(item['seq'], []).append((key, item))
    for seq, rows in by_frame.items():
        rows.sort(key=lambda row: row[1]['mask_bit'])
        data.frames[seq] = {'original_rgb_image': rows[0][1]['image']}
        for key, item in rows:
            i = item['mask_bit']
            item['mask_id'] = f'M{i+1:02d}'
            item.setdefault('overlay_color_rgb', [round(v*255) for v in colorsys.hsv_to_rgb((i*.61803398875)%1, .75, .95)])
    rendered = []
    originals = {str(p.relative_to(out)): digest(p) for p in (out/'evidence').glob('*.jpg')}
    for seq in sorted(by_frame):
        rendered.append(data.view(seq))
    for path in sorted((out/'evidence').glob('*.jpg')):
        if path.stem.startswith('frame_'):
            continue
        key = path.stem.replace('_', ':')
        if key not in items:
            raise ValueError(f'Unknown evidence: {path}')
        rendered.append(data.view(items[key]['seq'], key))
    save(data.out/'observations.json', items)
    page=['<!doctype html><meta charset="utf-8"><title>带 mask 编号的输入预览</title><style>body{max-width:1500px;margin:auto;padding:24px;font:18px/1.6 sans-serif}.pair{display:flex;gap:12px}.pair img{width:49%;object-fit:contain}pre{white-space:pre-wrap}</style><h1>带 mask 编号的输入预览</h1><p>由已有 mask 重绘，未重新调用 API。历史决策仍基于旧输入。M 编号只在当前帧内有效；帧序号与类别保留在图外。</p>']
    for seq, rows in sorted(by_frame.items()):
        rgb=html.escape(os.path.relpath(rows[0][1]['image'],data.out),quote=True)
        page.append(f'<h2>抽样帧 {seq} / 原始帧 {rows[0][1]["source_frame_id"]}</h2><div class="pair"><img loading="lazy" src="{rgb}"><img loading="lazy" src="evidence/frame_{seq:04d}.jpg"></div>')
        mapping=[{k:o.get(k) for k in ('id','mask_id','category_label','bbox_xyxy','overlay_color_rgb')} for _,o in rows]
        page.append('<details><summary>编号、类别与 bbox 对应关系</summary><pre>'+html.escape(json.dumps(mapping,ensure_ascii=False,indent=2))+'</pre></details>')
    (data.out/'index.html').write_text('\n'.join(page))
    save(data.out/'manifest.json', {'policy_version': 4, 'api_rerun': False,
         'note': 'New rendering from original masks; archived API requests and evidence are unchanged.',
         'original_evidence_sha256': originals,
         'rendered': {str(p.relative_to(data.out)): digest(p) for p in rendered}})
    result_path = out/'result_data.js'
    if result_path.exists():
        backup = data.out/'original_result_data.js'
        if not backup.exists(): shutil.copy2(result_path, backup)
        prefix = 'window.RESULT_FRAMES='
        rows = json.loads(result_path.read_text().removeprefix(prefix).rstrip(';\n'))
        for row in rows:
            filename = Path(row['agent_overlay']).name
            row['agent_overlay'] = 'numbered_overlay_preview/evidence/'+filename
            row['overlay_note'] = '编号 overlay 重绘预览；历史 Agent 决策未重跑'
        result_path.write_text(prefix+json.dumps(rows, ensure_ascii=False)+';')
        shutil.copy2(Path(__file__).parents[1]/'visualization/harness_input_display.js', out/'harness_input_display.js')
    assert originals == {str(p.relative_to(out)): digest(p) for p in (out/'evidence').glob('*.jpg')}
    print(json.dumps({'experiment': out.name, 'frames': len(by_frame), 'overlays': len(rendered), 'observations': len(items)}, ensure_ascii=False))


if __name__ == '__main__':
    for out in sys.argv[1:]: regenerate(out)
