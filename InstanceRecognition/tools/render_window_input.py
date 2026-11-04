"""Render one completed harness window's client input as a standalone page.

Everything here is read out of the run's own `calls/` log: the request that was
actually sent, not a reconstruction of it. The page lives outside the site tree
(it is a review artifact, not a published page), so image paths are relative to
the output file and point back into the run and the source.
"""
import argparse
import base64
import html
import json
import os
from pathlib import Path


def esc(text):
    return html.escape(str(text), quote=False)


def pretty(value):
    """Indented JSON, but short scalar arrays stay on one line.

    A plain json.dumps puts every bbox coordinate on its own line, which turns
    the 65-observation blocks into thousands of lines of single numbers and
    makes the page impossible to scan.
    """
    def scalar_list(item):
        if not isinstance(item, list) or not item:
            return False
        if any(isinstance(v, (dict, list)) for v in item):
            return False
        # Numeric arrays (bbox, centroid, axes) are short; ID strings can be long
        # lists but still read better on one wrapped line than 28 lines deep.
        return len(item) <= 8 or (all(isinstance(v, str) for v in item) and len(item) <= 48)

    def render(item, depth):
        pad = '  ' * depth
        if isinstance(item, dict):
            if not item:
                return '{}'
            rows = [f'{pad}  {json.dumps(k, ensure_ascii=False)}: {render(v, depth + 1)}'
                    for k, v in item.items()]
            return '{\n' + ',\n'.join(rows) + f'\n{pad}}}'
        if scalar_list(item):
            return '[' + ', '.join(json.dumps(v, ensure_ascii=False) for v in item) + ']'
        if isinstance(item, list):
            if not item:
                return '[]'
            rows = [f'{pad}  {render(v, depth + 1)}' for v in item]
            return '[\n' + ',\n'.join(rows) + f'\n{pad}]'
        return json.dumps(item, ensure_ascii=False)

    return render(value, 0)


def describe(index, block, seq_for_image=None):
    """One Chinese sentence saying what this block *is*, not what it starts with.

    The raw prefix of a 27 KB JSON tells a reader nothing they can act on, and
    every per-frame block starts with the same four tokens. The description is
    derived from the payload so it cannot drift from the content.
    """
    if block['type'] == 'image_url':
        kind = '原始 RGB 图' if seq_for_image is not None and seq_for_image[1] else '全帧 mask overlay'
        seq = seq_for_image[0] if seq_for_image else '?'
        return f'帧 {seq} 的{kind}，模型用来看真实外观与 mask 覆盖位置'
    text = block['text'].strip()
    if text.startswith('{'):
        try:
            parsed = json.loads(text)
        except ValueError:
            return '一个 JSON 文本块'
        if 'window' in parsed:
            objects = len(parsed['state']['objects'])
            return (f'窗口状态与全部历史：范围 {parsed["window"]}、版本 {parsed["state"]["version"]}、'
                    f'{objects} 个已归档物体（含 3D OBB）')
        if 'sampled_sequence' in parsed:
            n = len(parsed['observations'])
            return f'帧 {parsed["sampled_sequence"]} 的 {n} 条观测：类别、2D 框、点数、质心与 3 个空间候选'
        if 'history_access' in parsed:
            return '历史访问说明：告诉模型历史图片不随窗口发送，要看就调工具'
    if 'original RGB' in text:
        return '图注，声明下一张图是原始 RGB'
    if 'mask overlay' in text:
        return '图注，声明下一张图是全帧 mask 叠加图'
    return '一段固定说明文字'


def block_kind(block):
    return 'image' if block['type'] == 'image_url' else 'text'


def image_bytes(block):
    url = block['image_url']['url']
    if not url.startswith('data:image/'):
        return None
    payload = url.split(',', 1)[1]
    try:
        return base64.b64decode(payload)
    except (ValueError, IndexError):
        return None


def build(run_dir, first, output):
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve()
    request_path = sorted((run_dir / 'calls').glob(f'w{first:04d}_*_call00_request.json'))[0]
    checkpoint = json.loads((run_dir / 'checkpoint.json').read_text())
    config = json.loads((run_dir / 'config.json').read_text())
    request = json.loads(request_path.read_text())
    system, user = request['messages']
    blocks = user['content']

    # The logged request is redacted: image blocks carry a placeholder instead of
    # bytes. Resolve what was actually sent from the evidence the client encoded.
    evidence = run_dir / 'evidence'
    rgb_dir = Path(config['source']).resolve()
    frames_manifest = json.loads((rgb_dir / 'manifest.json').read_text())['frames']

    def ev_relative(path):
        return os.path.relpath(path, output.parent).replace(os.sep, '/')

    sections = []
    frame_images = {}
    text_chars = 0
    for index, block in enumerate(blocks):
        kind = block_kind(block)
        if kind == 'text':
            text_chars += len(block['text'])
            sections.append((index, 'text', len(block['text']), block['text']))
        else:
            sections.append((index, 'image', len(block['image_url']['url']), block))

    # Per-frame image files: block order is label, RGB, label, overlay. The
    # window covers a slice of the source, so the frame is looked up by the
    # window offset -- indexing the manifest directly would show frame 0 for the
    # window's first frame and silently caption every image with the wrong frame.
    by_sequence = {int(frame['sequence_index']): frame for frame in frames_manifest}
    frame_index = -1
    image_seq = 0
    rows = []
    for index, block in enumerate(blocks):
        if block['type'] != 'image_url':
            continue
        image_seq += 1
        is_rgb = image_seq % 2 == 1
        if is_rgb:
            frame_index += 1
        rows.append((index, block, first + frame_index, is_rgb))

    image_html = {}
    total_pixels = 0
    for index, block, seq, is_rgb in rows:
        frame = by_sequence.get(seq)
        if frame is None:
            continue
        if is_rgb:
            path = (rgb_dir / frame['original_rgb_image']).resolve()
            label = f"帧 {frame['sequence_index']} · 原始 RGB（源帧 {frame['source_frame_id']}）"
        else:
            path = evidence / f"frame_{frame['sequence_index']:04d}.jpg"
            label = f"帧 {frame['sequence_index']} · 全帧 mask overlay（源帧 {frame['source_frame_id']}）"
        if not path.is_file():
            continue
        total_pixels += path.stat().st_size
        image_html[index] = (
            f'<figure class="shot"><img loading="lazy" src="{esc(ev_relative(path))}" alt="{esc(label)}">'
            f'<figcaption><b>{esc(label)}</b><br><code>{esc(str(path.name))}</code> · '
            f'{path.stat().st_size // 1024} KB</figcaption></figure>')

    # Block outline, then the details. Images are described by the frame and kind
    # resolved above, so the outline and the figure captions cannot disagree.
    image_slot = {index: (seq, is_rgb) for index, _block, seq, is_rgb in rows}
    parts = []
    parts.append('<ol class="outline">')
    for index, kind, size, payload in sections:
        sentence = describe(index, blocks[index], image_slot.get(index))
        parts.append(f'<li><span class="ix">{index:02d}</span>'
                     f'<span class="tag {kind}">{kind}</span>'
                     f'<span class="nm">{esc(sentence)}</span>'
                     f'<span class="sz">{"1 张图" if kind == "image" else f"{size:,} 字符"}</span></li>')
    parts.append('</ol>')

    # The body follows the message itself: envelope, then the system message,
    # then the user message -- its outline, then one section per block in the
    # order the blocks actually appear. A section number must never suggest a
    # block belongs to a message it is not part of.
    state_text = blocks[0]['text']
    state = json.loads(state_text)
    observation_count = len(state['state']['objects'])

    body = []
    body.append(f"""<section><h2><span class="no">0</span>请求信封</h2>
<pre class="json">{esc(pretty({k: request[k] for k in request if k not in ('messages', 'tools')}))}</pre>
<p>工具定义共 {len(request['tools'])} 个：{esc('、'.join(t['function']['name'] for t in request['tools']))}。
完整定义见运行目录的 <code>{esc(request_path.name)}</code>。</p></section>""")

    body.append(f"""<section><h2><span class="no">1</span>system 消息</h2>
<p>整个 system 消息只有这一份固定指令，每个窗口完全相同，不含任何观测、物体或图片。
它分四节：术语与输入、每窗工作流、工具与证据、事务协议。</p>
<dl class="facts"><dt>来源</dt><dd><code>src/harness/prompt.txt</code></dd>
<dt>长度</dt><dd>{len(system['content']):,} 字符</dd>
<dt>校验</dt><dd><code>prompt_sha256 = {esc(config.get('prompt_sha256', '?'))[:16]}…</code></dd></dl>
<details><summary>展开全文</summary><pre class="prompt">{esc(system['content'])}</pre></details></section>""")

    body.append(f"""<section><h2><span class="no">2</span>user 消息 · 27 个内容块</h2>
<p>一条 user 消息，按顺序拼装。文本合计 <b>{text_chars:,}</b> 字符；图片 10 张，
日志里以 33 字符占位符脱敏（下方「实际图片」还原了这 10 张）。</p>
{''.join(parts)}</section>""")

    # Block 00 is the largest and carries the whole history, so it gets a
    # summary and labelled examples instead of one 27 KB dump.
    body.append(f"""<section>
<h3>block 00 · 窗口状态与全部历史 <span class="h">{len(state_text):,} 字符</span></h3>
<p>模型在开头拿到的就是这一块：窗口范围、版本、{observation_count} 个已归档物体的完整档案、
{len(state['global_object_geometry']['objects'])} 个 3D OBB。</p>
<dl class="facts">
<dt>window</dt><dd>{esc(state['window'])}</dd>
<dt>state.version</dt><dd>{state['state']['version']}</dd>
<dt>state.objects</dt><dd>{observation_count} 个已归档物体（每个含标签、观测数、代表观测 id 与 OBB）</dd>
<dt>global_object_geometry</dt><dd>{len(state['global_object_geometry']['objects'])} 个直立 OBB（中心、尺寸、三轴朝向）</dd>
<dt>max_calls</dt><dd>{state['max_calls']}</dd>
</dl>
<h3>其中一个物体档案</h3>
<pre class="json">{esc(pretty(state['state']['objects'][0]))}</pre>
<h3>同一个物体的 OBB</h3>
<pre class="json">{esc(pretty(state['global_object_geometry']['objects'][0]))}</pre>
<details><summary>展开 block 00 全文</summary><pre class="json">{esc(pretty(state))}</pre></details>
</section>""")

    # One section per remaining block, in block order.
    for index, kind, size, payload in sections:
        if index == 0:
            continue
        if kind == 'image':
            slot = image_slot.get(index)
            seq, is_rgb = slot if slot else (None, True)
            body.append(f"""<section class="slim"><h3>block {index:02d} · 图片</h3>
<p class="mono">{esc(describe(index, blocks[index], slot))}。实际文件见下方「实际图片」中的帧 {seq} 条目。</p></section>""")
            continue
        if size < 200:
            body.append(f"""<section class="slim"><h3>block {index:02d} · 图注</h3>
<p class="mono">{esc(payload)}</p></section>""")
            continue
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = None
        if parsed is not None and 'sampled_sequence' in parsed:
            rendered = pretty(parsed)
            title = f"block {index:02d} · 帧 {parsed['sampled_sequence']} 的观测记录"
            summary = f" · {len(parsed['observations'])} 条观测"
        else:
            rendered = pretty(parsed) if parsed is not None else payload
            title = f"block {index:02d} · 历史访问说明"
            summary = ''
        body.append(f"""<section><h3>{esc(title)} <span class="h">{size:,} 字符{summary}</span></h3>
<pre class="json">{esc(rendered)}</pre></section>""")

    shots = ''.join(image_html[i] for i in sorted(image_html))
    body.append(f"""<section><h2><span class="no">3</span>实际图片</h2>
<p>请求日志对图片做了脱敏，这里按客户端当时编码的同一份文件还原：原始 RGB 来自 source 的
<code>original_rgb_image</code>，overlay 来自运行目录的 <code>evidence/</code>。
10 张合计约 {total_pixels // 1024} KB。</p>
<div class="shots">{shots}</div></section>""")
    page = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harness 单窗口输入 · 窗口 8（帧 35–39）</title>
<style>
:root{{--ink:#16191d;--ink2:#454c55;--ink3:#6b7480;--line:#dfe3e8;--line2:#eef1f4;
--bg:#fbfcfd;--card:#fff;--accent:#0f766e;--accent2:#e6f4f1;--warn:#b45309;
--mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1040px;margin:0 auto;padding:0 28px 120px}}
header{{padding:52px 0 26px;border-bottom:1px solid var(--line)}}
.eyebrow{{font:11.5px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:12px}}
h1{{margin:0 0 12px;font-size:32px;line-height:1.25;letter-spacing:-.02em}}
.lede{{margin:0;color:var(--ink2);max-width:64ch}}
.meta{{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px}}
.tag{{font:11.5px var(--mono);background:var(--card);border:1px solid var(--line);border-radius:5px;padding:3px 9px;color:var(--ink2)}}
h2{{margin:44px 0 10px;font-size:21px;display:flex;align-items:baseline;gap:10px}}
h2 .no{{font:13px var(--mono);color:var(--accent);font-weight:600}}
h3{{margin:26px 0 8px;font-size:15.5px}}
h3 .h{{font:12px var(--mono);color:var(--ink3);font-weight:400;margin-left:6px}}
p{{margin:0 0 12px}}
code{{font:12.8px var(--mono);background:var(--line2);padding:1.5px 5px;border-radius:4px}}
pre{{margin:10px 0 16px;padding:13px 15px;background:#1d2228;color:#dfe6ee;border-radius:8px;
overflow:auto;font:12.4px/1.6 var(--mono);max-height:460px}}
pre.prompt{{white-space:pre-wrap;max-height:520px}}
pre.json{{white-space:pre-wrap;word-break:break-word}}
pre.plain{{white-space:pre-wrap;max-height:none}}
details{{margin:10px 0 18px}}
summary{{cursor:pointer;font-size:13.5px;color:var(--accent);font-weight:600;padding:6px 0}}
ol.outline{{list-style:none;padding:0;margin:14px 0 20px;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--card)}}
ol.outline li{{display:grid;grid-template-columns:38px 62px minmax(0,1fr) 96px;gap:10px;align-items:center;
padding:7px 14px;border-bottom:1px solid var(--line2);font-size:13.5px}}
ol.outline li:last-child{{border-bottom:none}}
.ix{{font:12px var(--mono);color:var(--ink3)}}
.tag.text,.tag.image{{font:11px var(--mono);text-align:center;border-radius:4px;padding:1px 6px}}
.tag.text{{background:var(--accent2);color:var(--accent)}}
.tag.image{{background:#f4f0fd;color:#6d28d9}}
.nm{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink2)}}
.sz{{font:11.5px var(--mono);color:var(--ink3);text-align:right}}
dl.facts{{display:grid;grid-template-columns:minmax(150px,auto) minmax(0,1fr);gap:6px 16px;margin:12px 0 18px;font-size:14px}}
dl.facts dt{{font:11px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);padding-top:4px;white-space:nowrap}}
dl.facts dd{{margin:0}}
.mono{{font:12.5px var(--mono);color:var(--ink2);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px}}
section.slim h3{{margin-bottom:4px}}
.shots{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-top:16px}}
.shot{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:9px;overflow:hidden}}
.shot img{{width:100%;display:block;background:#12161b}}
.shot figcaption{{padding:9px 12px;font-size:12.5px;color:var(--ink3);line-height:1.6}}
.note{{border-left:3px solid var(--warn);background:#fef6e7;padding:12px 16px;border-radius:0 7px 7px 0;margin:14px 0;font-size:14px}}
</style></head><body><div class="wrap">
<header>
<div class="eyebrow">Harness · single window input</div>
<h1>窗口 8（帧 35–39）的模型输入</h1>
<p class="lede">这是 2026-10-16 那次 150 帧运行里第 8 个窗口<strong>实际发出</strong>的请求。
所有内容从 <code>calls/</code> 的请求日志读取，未做改写；图片从客户端当时编码的同一份文件还原。</p>
<div class="meta">
<span class="tag">run <b>harness-v5-amb3r-150frames-20261016-1550</b></span>
<span class="tag">model <b>{esc(request['model'])}</b></span>
<span class="tag">窗口 <b>[35, 39]</b></span>
<span class="tag">base version <b>7</b></span>
<span class="tag">prompt tokens <b>53,453</b></span>
</div>
</header>
{''.join(body)}
</div></body></html>'''
    output.write_text(page, encoding='utf-8')
    print(f'wrote {output} ({len(page):,} chars)')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--window-start', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.run, args.window_start, args.output)


if __name__ == '__main__':
    main()
