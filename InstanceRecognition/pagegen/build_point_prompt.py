"""Results page of a point-prompt object run: one frame at a time, beside the text-prompt masks.

    python -m pagegen.build_point_prompt --run point-full100_v2 --out point-prompt-full100-v2.html
    python -m pagegen.build_point_prompt --run point-full100_v2 --left point-full100_v1   # compare two runs

Writes docs/<out>. Images are downsized and copied into docs/assets/point-prompt/<run>/ so
the page does not depend on the run directory. The GT columns (merged masks, the cabinet /
bookcase outcome) come from tools.point_prompt and are evaluation only.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from src import config
from src.perception.point_prompt.objects import run_dir_of

DOCS = config.ROOT / 'docs'
WIDTH = 720
TEXT = config.ARTIFACTS_ROOT / 'experiments/gpt_sam31_uniform500/perception/masks'
MIN_AREA = 0.005


def text_side(fid):
    p = json.loads((TEXT / f'frame_{fid:06d}/sam31/predictions.json').read_text())
    W, H = p['image_shape_hwc'][1], p['image_shape_hwc'][0]
    inst = [i for i in p['instances'] if i['mask_pixels'] >= MIN_AREA * W * H]
    counts = {}
    for i in inst:
        counts[i['class_name']] = counts.get(i['class_name'], 0) + 1
    return TEXT / f'frame_{fid:06d}/sam31/annotated.jpg', len(inst), counts




def shrink(src, dst):
    if dst.exists():
        return
    im = Image.open(src).convert('RGB')
    im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.Resampling.LANCZOS)
    im.save(dst, quality=80)


def left_side(fid, left, assets, rel):
    """The left column: the text pipeline, or another point-prompt run to compare against."""
    if left == 'text':
        img, n, counts = text_side(fid)
        shrink(img, assets / f'{fid}_text.jpg')
        return f'{rel}/{fid}_text.jpg', n, counts
    folder = run_dir_of(left) / f'frame_{fid:06d}'
    r = json.loads((folder / 'result.json').read_text())
    left_assets = DOCS / 'assets/point-prompt' / left.removeprefix('point-')
    left_assets.mkdir(parents=True, exist_ok=True)
    shrink(folder / 'kept.jpg', left_assets / f'{fid}_point.jpg')
    return (f'assets/point-prompt/{left.removeprefix("point-")}/{fid}_point.jpg', len(r['kept']),
            dict(Counter(k['label'] for k in r['kept'])))


def frame_record(folder, assets, rel, left='text'):
    r = json.loads((folder / 'result.json').read_text())
    fid = r['frame_id']
    limg, tn, tcounts = left_side(fid, left, assets, rel)
    shrink(folder / 'kept.jpg', assets / f'{fid}_point.jpg')
    steps = []
    for call in sorted((folder / 'calls').glob('r*_tool.json')):
        t = json.loads(call.read_text())
        res = t['result']
        note = ''
        if 'error' in res:
            note = 'error: ' + res['error'][:80]
        elif t['name'] == 'plan':
            note = f'+{len(res.get("planned", []))}'
        elif t['name'] == 'segment':
            c = res.get('candidates', [])
            w = sum(len(x['warnings']) for x in c)
            note = f'{len(c)}' + (f', {w}⚠' if w else '')
        elif t['name'] == 'keep':
            note = f'{len(res.get("kept", []))}'
        elif t['name'] == 'combine':
            note = res.get('plan_id', '')
        steps.append([t['name'], note])
    return {
        'fid': fid, 'rounds': r['rounds'], 'seconds': r['seconds'],
        'tok': r['usage']['prompt_tokens'],
        'kept': [{'pid': k['plan_id'], 'label': k['label'], 'where': k['where'],
                  'pos': len(k['positive']), 'neg': len(k['negative']),
                  'area': round(k['area_fraction'], 4), 'score': round(k['sam_score'], 3),
                  'combined': bool(k.get('combined_from'))} for k in r['kept']],
        'skipped': [{'pid': s['plan_id'], 'label': s['label'], 'reason': s['reason']} for s in r['skipped']],
        'text_n': tn, 'text_counts': tcounts, 'text_max': max(tcounts.values()) if tcounts else 0,
        'steps': steps,
        'img_text': limg, 'img_point': f'{rel}/{fid}_point.jpg',
    }


def gt_columns(run, left, fids):
    """Per frame, scored against ScanNet GT (evaluation only; the agent never sees GT):
    GT-merged masks on each side, the right side's merged pairs, and the
    cabinet#8 / shelf#56 outcome on each side."""
    import zipfile
    from tools.point_prompt import gt_eval as G
    from tools.point_prompt.compare_runs import classify
    right = dict(G.point_masks(run))
    lft = dict(G.text_masks()) if left == 'text' else dict(G.point_masks(left))
    archive = zipfile.ZipFile(f'{G.SCENE}_2d-instance.zip')
    pr = G.evaluate([(f, right[f]) for f in fids])[3]
    pl = G.evaluate([(f, lft[f]) for f in fids if f in lft])[3]
    return {f: {'m_left': len(pl.get(f, [])), 'm_right': len(pr.get(f, [])),
                'm_pairs': [' + '.join(f'{p[1]}#{p[0]}' for p in r['gt']) for r in pr.get(f, [])],
                'cab': [classify(f, lft[f], archive) if f in lft else 'n/a',
                        classify(f, right[f], archive)]}
            for f in fids}


# The yellow box above the viewer, per run. Runs without an entry get none.
NOTES = {
    'full100_v1': '<b>粒度在不同运行间不稳定。</b>同一组连在一起的嵌入式柜子，有时被分成几件、有时合成一件：'
                  '帧 2571 在 trial3 里是电视柜 / 墙柜 / 高柜三件，这次是一个占画面 62% 的 <code>wall unit</code>。'
                  'prompt 目前没有规定这种情况怎么切。用“大物体”筛选可以集中看这类帧。',
    'full100_v2': '<b>这次的 prompt 改了物体边界规则。</b>两件家具是否分开，只看结构断口：缝、立柱、侧板、'
                  '进深或高度台阶、转角、材料变化；开间里放的是书、门还是电视不算。电视柜和书柜墙在两者都清楚可见的 13 帧里'
                  '有 12 帧分开（上一版 11 帧合成一个 mask）。代价是书柜墙在 5 帧里被切成 2–3 块、电视柜的高柜侧翼在 3 帧里被单独切出，'
                  '这类拆分下游可以再合并。用“柜子帧”筛选可以集中看。',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True)
    ap.add_argument('--left', default='text', help="left column: 'text' (current pipeline) or another run")
    ap.add_argument('--out', default='point-prompt-full100.html', help='file name under docs/')
    args = ap.parse_args()
    run = run_dir_of(args.run)
    name = args.run.removeprefix('point-')            # the published asset folders predate the point- prefix
    assets = DOCS / 'assets/point-prompt' / name
    assets.mkdir(parents=True, exist_ok=True)
    rel = f'assets/point-prompt/{name}'
    frames = [frame_record(d, assets, rel, args.left)
              for d in sorted(run.glob('frame_*')) if (d / 'result.json').exists()]
    gt = gt_columns(args.run, args.left, [f['fid'] for f in frames])
    for f in frames:
        f.update(gt[f['fid']])
    summary = json.loads((run / 'summary.json').read_text())
    cfg = json.loads((run / 'config.json').read_text())
    labels = Counter(k['label'] for f in frames for k in f['kept'])
    data = {'run': name, 'summary': summary, 'model': cfg['model'],
            'prompt_sha': cfg['prompt_sha256'][:8], 'frames': frames,
            'labels': labels.most_common(40), 'left': args.left.removeprefix('point-'),
            'left_label': '现行' if args.left == 'text' else args.left.removeprefix('point-'),
            'note': NOTES.get(name, '')}

    head = (DOCS / 'algorithm-1105-v1.html').read_text()
    head = head[:head.index('</style>')].replace(
        '<title>流水线设计 · 1105-v1 · InstanceRecognition</title>',
        f'<title>点选分割 100 帧结果 · {name} · InstanceRecognition</title>')
    page = head + CSS + BODY.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    out = DOCS / args.out
    out.write_text(page)
    print(out, len(frames), 'frames')


CSS = """
  .wrap{max-width:1500px; grid-template-columns:minmax(0,1fr) !important}
  .stats{display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px; margin:18px 0}
  .stat{background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px}
  .stat .v{font-size:24px; font-weight:700; letter-spacing:-.01em}
  .stat .k{font-size:12px; color:var(--ink-3)}
  .stat .s{font-size:12px; color:var(--ink-2); font-family:var(--mono)}
  .bar{position:sticky; top:0; z-index:5; background:rgba(251,252,253,.96); backdrop-filter:blur(4px);
       border-bottom:1px solid var(--line); padding:10px 0; margin:0 0 14px; display:flex; flex-wrap:wrap; gap:8px; align-items:center}
  .bar button{font:inherit; font-size:13px; padding:5px 11px; border-radius:16px; border:1px solid var(--line); background:#fff; cursor:pointer; color:var(--ink-2)}
  .bar button.on{background:var(--accent); color:#fff; border-color:var(--accent)}
  .bar select,.bar input{font:inherit; font-size:13px; padding:4px 8px; border:1px solid var(--line); border-radius:6px; background:#fff}
  .bar .count{margin-left:auto; font-family:var(--mono); font-size:12.5px; color:var(--ink-3)}
  .card{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:0 0 16px}
  .card h3{margin:0 0 8px; font-size:16px; display:flex; flex-wrap:wrap; gap:8px; align-items:baseline}
  .card h3 .m{font-family:var(--mono); font-size:12px; color:var(--ink-3); font-weight:500}
  .chip{font-family:var(--mono); font-size:11px; padding:1px 7px; border-radius:10px; font-weight:600}
  .chip.frag{background:var(--stop-soft); color:var(--stop)} .chip.big{background:var(--warn-soft); color:var(--warn)}
  .chip.comb{background:var(--model-soft); color:var(--model)} .chip.slow{background:#eef4ff; color:#1d4ed8}
  .pair2{display:grid; grid-template-columns:1fr 1fr; gap:12px}
  .pair2 img{width:100%; border:1px solid var(--line); border-radius:8px; display:block; cursor:zoom-in}
  .pair2 .lbl{font-family:var(--mono); font-size:11px; font-weight:700; letter-spacing:.06em; margin-bottom:4px}
  .lbl.old{color:var(--stop)} .lbl.new{color:var(--accent)}
  .cap{font-size:12.5px; color:var(--ink-2); margin-top:5px; line-height:1.55}
  details{margin-top:8px; font-size:13px}
  details summary{cursor:pointer; color:var(--accent); font-size:13px}
  .kt{width:100%; font-size:12.5px; margin:6px 0}
  .kt td,.kt th{padding:3px 8px}
  .steps{font-family:var(--mono); font-size:12px; color:var(--ink-2); line-height:1.9}
  .steps span{display:inline-block; background:var(--line-2); border-radius:4px; padding:0 6px; margin:0 4px 3px 0}
  .steps span.e{background:var(--stop-soft); color:var(--stop)}
  .nav{position:sticky; top:0; z-index:5; background:rgba(251,252,253,.97); backdrop-filter:blur(4px);
       border:1px solid var(--line); border-radius:12px; padding:12px 16px; margin:0 0 14px}
  #bar button{font:inherit; font-size:13px; padding:4px 11px; border-radius:16px; border:1px solid var(--line); background:#fff; cursor:pointer; color:var(--ink-2)}
  #bar button.on{background:var(--accent); color:#fff; border-color:var(--accent)}
  #bar input{font:inherit; font-size:13px; padding:4px 8px; border:1px solid var(--line); border-radius:6px; background:#fff}
  .nav .goto{font:inherit; font-size:13px; padding:4px 8px; border:1px solid var(--line); border-radius:6px}
  .nav .row{display:flex; flex-wrap:wrap; gap:10px; align-items:center}
  .nav .row + .row{margin-top:10px}
  .nav .ftitle{font-size:20px; font-weight:700; min-width:118px}
  .nav .pos{font-family:var(--mono); font-size:12.5px; color:var(--ink-3)}
  .nav button.step{font:inherit; font-size:16px; width:34px; height:30px; border-radius:7px; border:1px solid var(--line); background:#fff; cursor:pointer}
  .nav button.step:disabled{opacity:.35; cursor:default}
  .nav input[type=range]{flex:1; min-width:280px; accent-color:var(--accent)}
  .nav .goto{width:72px}
  .strip{display:flex; gap:2px; margin-top:8px; height:16px; padding:2px 0; box-sizing:content-box}
  .strip i{flex:1 1 0; min-width:0; background:#dfe3e8; border-radius:2px; cursor:pointer; position:relative}
  .strip i.sf{background:#f3a7a7} .strip i.sb{background:#f2c879} .strip i.sf.sb{background:linear-gradient(#f3a7a7 50%,#f2c879 50%)}
  .strip i.cur{outline:2px solid var(--ink); outline-offset:1px; z-index:1}
  .strip i:hover{filter:brightness(.85)}
  .hint{font-size:12px; color:var(--ink-3)}
  .hint b{display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:-1px; margin:0 3px 0 8px}
  .card .pair2 img{max-height:72vh; object-fit:contain; background:#000}
  .lb{position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center; justify-content:center; z-index:50; cursor:zoom-out}
  .lb img{max-width:96vw; max-height:94vh}
  .labels{font-size:12.5px; line-height:2}
  .labels span{display:inline-block; background:var(--line-2); border-radius:4px; padding:0 7px; margin:0 4px 0 0}
</style>
</head>
"""

BODY = r"""<body>
<div class="wrap">
<main>
<header class="top" style="padding-bottom:20px">
  <div class="eyebrow">InstanceRecognition · Perception · perception_agent 100 帧结果</div>
  <h1>GPT 点选 + SAM3：100 帧自动运行结果</h1>
  <p class="lede">与当前 harness 实验同一组 100 帧（<code>scene_amb3r_full100_v1</code>，覆盖整段录像）。
  每次显示一帧：<span id="leftdesc"></span>右边是 perception_agent 本次运行保留的 mask。用进度条、‹ › 按钮或 ←/→ 键切换，点图片可放大。</p>
  <div class="meta" id="meta"></div>
</header>

<div class="stats" id="stats"></div>

<div class="warn" id="note"></div>

<details style="margin:0 0 14px"><summary>保留 mask 的类别分布（前 40）</summary><div class="labels" id="labels"></div></details>

<div class="nav" id="nav">
  <div class="row" id="bar">
    <span class="hint" style="margin-right:2px">浏览范围</span>
    <button data-f="all" class="on">全部</button>
    <button data-f="frag" id="fragbtn">现行有碎片（同类 ≥5）</button>
    <button data-f="merge">右边有 GT 合并</button>
    <button data-f="cab">柜子帧（电视柜+书柜墙都可见）</button>
    <button data-f="big">含大物体（&gt;40% 画面）</button>
    <button data-f="comb">用了 combine</button>
    <button data-f="skip">有跳过</button>
    <button data-f="slow">轮数 ≥10</button>
    <input id="q" placeholder="搜类别，如 cabinet" size="16">
  </div>
  <div class="row">
    <span class="ftitle" id="title">帧 —</span>
    <button class="step" id="prev" title="上一帧（←）">‹</button>
    <input type="range" id="slider" min="0" max="0" value="0">
    <button class="step" id="next" title="下一帧（→）">›</button>
    <input class="goto" id="goto" placeholder="帧号">
    <span class="pos" id="pos"></span>
  </div>
  <div class="strip" id="strip"></div>
  <div class="hint" style="margin-top:6px">每格一帧，点击跳转；←/→ 键切换。
    <span id="fraglegend"><b style="background:#f3a7a7"></b>现行有碎片</span> <b style="background:#f2c879"></b>含大物体 <b style="background:#dfe3e8"></b>其他
    · GT 只用于评估，不给 agent 看</div>
</div>

<div id="view"></div>
<footer><p id="foot"></p></footer>
</main>
</div>
<div class="lb" id="lb"><img id="lbimg"></div>

<script>
const D = __DATA__;
const S = D.summary, F = D.frames;
const n = F.length;
const med = a => { const b=[...a].sort((x,y)=>x-y); return b[Math.floor(b.length/2)]; };
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const textTotal = F.reduce((a,f)=>a+f.text_n,0);
const fragN = F.filter(f=>f.text_max>=5).length;

const isText = D.left === 'text';
document.getElementById('leftdesc').innerHTML = isText
  ? '左边是现行流程（GPT 类别 + SAM 文本，面积过滤后），'
  : `左边是上一版运行 <code>${esc(D.left)}</code>，`;
const note = document.getElementById('note');
if (D.note) note.innerHTML = D.note; else note.remove();
if (!isText) { document.getElementById('fragbtn').remove(); document.getElementById('fraglegend').remove(); }
const CAB = {merged:'合成一个', separated:'分开', partial:'部分', 'n/a':''};
const mergeL = F.reduce((a,f)=>a+f.m_left,0), mergeR = F.reduce((a,f)=>a+f.m_right,0);
const cabSep = F.filter(f=>f.cab[1]==='separated').length, cabMerL = F.filter(f=>f.cab[0]==='merged').length,
      cabMerR = F.filter(f=>f.cab[1]==='merged').length, cabN = F.filter(f=>f.cab[1]!=='n/a').length;
document.getElementById('meta').innerHTML = [
  ['运行', D.run], ['GPT', D.model], ['prompt', D.prompt_sha], ['SAM', 'sam3.pt · 交互式点 · 3 遍细化'],
].map(([k,v])=>`<span class="tag">${k} <b>${esc(v)}</b></span>`).join('');

document.getElementById('stats').innerHTML = [
  [`${S.finished} / ${S.frames}`, '完成帧', `出错 ${S.error.length} · 未完成 ${S.unfinished}`],
  [S.kept, '保留 mask', `${D.left_label} ${textTotal}`],
  [mergeR, 'GT 合并 mask', `${D.left_label} ${mergeL} · 一个 mask 盖住 ≥2 件 GT 物体`],
  [`${cabSep} / ${cabN}`, '电视柜与书柜墙分开', `合成一个：${D.left_label} ${cabMerL} 帧 → 本次 ${cabMerR} 帧`],
  [S.skipped, '跳过', '多数因低于面积门槛'],
  [med(F.map(f=>f.rounds)), '轮数中位', `平均 ${(S.rounds/n).toFixed(1)} · 最多 ${Math.max(...F.map(f=>f.rounds))}`],
  [(S.prompt_tokens/1e6).toFixed(1)+'M', 'prompt token', `每帧中位 ${(med(F.map(f=>f.tok))/1e3).toFixed(0)}k`],
  [Math.round(S.this_invocation_seconds/60)+' min', '总耗时', `4 帧并行 · 每帧中位 ${med(F.map(f=>f.seconds)).toFixed(0)} s`],
  ...(isText ? [[fragN, '现行有碎片的帧', '同一类别 ≥5 个 mask']] : []),
].map(([v,k,s])=>`<div class="stat"><div class="v">${v}</div><div class="k">${k}</div><div class="s">${s}</div></div>`).join('');

document.getElementById('labels').innerHTML = D.labels.map(([l,c])=>`<span>${esc(l)} ${c}</span>`).join('');
document.getElementById('foot').textContent = `InstanceRecognition · perception_agent · ${D.run} · ${n} 帧`;

const tags = f => ({
  frag: isText && f.text_max >= 5,
  merge: f.m_right > 0,
  cab: f.cab[1] !== 'n/a',
  big: f.kept.some(k=>k.area>0.4),
  comb: f.kept.some(k=>k.combined),
  skip: f.skipped.length > 0,
  slow: f.rounds >= 10,
});

function card(f){
  const t = tags(f);
  const chips = [
    t.frag && `<span class="chip frag">现行碎片 ×${f.text_max}</span>`,
    t.big && `<span class="chip big">大物体 ${Math.round(Math.max(...f.kept.map(k=>k.area))*100)}%</span>`,
    t.comb && `<span class="chip comb">combine</span>`,
    t.slow && `<span class="chip slow">${f.rounds} 轮</span>`,
    t.merge && `<span class="chip frag" title="${esc(f.m_pairs.join('；'))}">GT 合并 ×${f.m_right}：${esc(f.m_pairs.join('；'))}</span>`,
    t.cab && `<span class="chip comb">电视柜/书柜墙：${CAB[f.cab[0]]||'—'} → ${CAB[f.cab[1]]||'—'}</span>`,
  ].filter(Boolean).join('');
  const tc = Object.entries(f.text_counts).sort((a,b)=>b[1]-a[1]).map(([k,v])=> v>1?`${esc(k)}×${v}`:esc(k)).join('，');
  const kc = {}; f.kept.forEach(k=>kc[k.label]=(kc[k.label]||0)+1);
  const pc = Object.entries(kc).sort((a,b)=>b[1]-a[1]).map(([k,v])=> v>1?`${esc(k)}×${v}`:esc(k)).join('，');
  const rows = f.kept.map(k=>`<tr><td>${k.pid}</td><td>${esc(k.label)}${k.combined?' <span class="chip comb">C</span>':''}</td>
     <td>${esc(k.where)}</td><td class="num">+${k.pos} / −${k.neg}</td><td class="num">${(k.area*100).toFixed(1)}%</td><td class="num">${k.score}</td></tr>`).join('');
  const sk = f.skipped.length ? `<p style="margin:10px 0 2px"><b>跳过</b></p><ul style="margin:0">${f.skipped.map(s=>`<li>${s.pid} ${esc(s.label)} — <span style="color:var(--ink-3)">${esc(s.reason)}</span></li>`).join('')}</ul>` : '';
  const steps = f.steps.map((s,i)=>`<span class="${s[1].startsWith('error')?'e':''}">r${i} ${s[0]}${s[1]?' '+esc(s[1]):''}</span>`).join('');
  return `<div class="card">
    <h3>帧 ${f.fid} ${chips}<span class="m">${f.rounds} 轮 · ${(f.tok/1e3).toFixed(0)}k tok · ${f.seconds.toFixed(0)} s</span></h3>
    <div class="pair2">
      <div><div class="lbl old">${esc(D.left_label)} · ${f.text_n} 个${f.m_left?` · GT 合并 ${f.m_left}`:''}</div><img src="${f.img_text}"><div class="cap">${tc||'—'}</div></div>
      <div><div class="lbl new">${esc(D.run)} · ${f.kept.length} 个${f.skipped.length?` · 跳过 ${f.skipped.length}`:''}</div><img src="${f.img_point}"><div class="cap">${pc}</div></div>
    </div>
    <p style="margin:12px 0 2px"><b>保留明细</b></p>
    <table class="kt"><thead><tr><th>计划</th><th>类别</th><th>位置描述（GPT 写的）</th><th>点</th><th>面积</th><th>SAM</th></tr></thead><tbody>${rows}</tbody></table>
    ${sk}
    <p style="margin:10px 0 2px"><b>调用序列</b> <span class="hint">segment 后的数字为候选数，⚠ 为警告数</span></p><div class="steps">${steps}</div>
  </div>`;
}

let filt = 'all', list = F, idx = 0;
const slider = document.getElementById('slider');

function rebuild(keepFid){
  const q = document.getElementById('q').value.trim().toLowerCase();
  list = F.filter(f => filt==='all' || tags(f)[filt]);
  if (q) list = list.filter(f => f.kept.some(k=>k.label.toLowerCase().includes(q)) || Object.keys(f.text_counts).some(k=>k.toLowerCase().includes(q)));
  slider.max = Math.max(0, list.length-1);
  const strip = document.getElementById('strip');
  strip.innerHTML = list.map((f,i)=>{ const t=tags(f); return `<i data-i="${i}" class="${t.frag?'sf':''} ${t.big?'sb':''}" title="帧 ${f.fid}"></i>`; }).join('');
  const j = list.findIndex(f=>f.fid===keepFid);
  show(j >= 0 ? j : 0);
}

function show(i){
  const view = document.getElementById('view');
  if (!list.length){
    view.innerHTML = '<div class="card">没有符合条件的帧。</div>';
    document.getElementById('title').textContent = '帧 —';
    document.getElementById('pos').textContent = '0 / 0';
    return;
  }
  idx = Math.max(0, Math.min(list.length-1, i));
  const f = list[idx];
  view.innerHTML = card(f);
  slider.value = idx;
  document.getElementById('title').textContent = `帧 ${f.fid}`;
  document.getElementById('pos').textContent = `第 ${idx+1} / ${list.length} 帧（共 ${n}）`;
  document.getElementById('prev').disabled = idx === 0;
  document.getElementById('next').disabled = idx === list.length-1;
  document.querySelectorAll('#strip i').forEach((c,k)=>c.classList.toggle('cur', k===idx));
  history.replaceState(null, '', '#f' + f.fid);
  // preload neighbours so stepping is instant
  [idx-1, idx+1].filter(k=>k>=0 && k<list.length).forEach(k=>{ new Image().src=list[k].img_text; new Image().src=list[k].img_point; });
}

document.querySelectorAll('#bar button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#bar button').forEach(x=>x.classList.remove('on')); b.classList.add('on');
  filt = b.dataset.f; rebuild(list[idx] && list[idx].fid);
});
document.getElementById('q').oninput = ()=> rebuild(list[idx] && list[idx].fid);
slider.oninput = ()=> show(+slider.value);
document.getElementById('prev').onclick = ()=> show(idx-1);
document.getElementById('next').onclick = ()=> show(idx+1);
document.getElementById('strip').addEventListener('click', e=>{ if (e.target.dataset.i !== undefined) show(+e.target.dataset.i); });
document.getElementById('goto').addEventListener('keydown', e=>{
  if (e.key !== 'Enter') return;
  const fid = +e.target.value; let j = list.findIndex(f=>f.fid===fid);
  if (j < 0){ // not in the current filter: nearest frame number
    j = list.reduce((best,f,k)=> Math.abs(f.fid-fid) < Math.abs(list[best].fid-fid) ? k : best, 0);
  }
  show(j); e.target.blur();
});
document.addEventListener('keydown', e=>{
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.key === 'ArrowLeft'){ show(idx-1); e.preventDefault(); }
  else if (e.key === 'ArrowRight'){ show(idx+1); e.preventDefault(); }
  else if (e.key === 'Escape') lb.style.display = 'none';
});
const lb = document.getElementById('lb');
document.getElementById('view').addEventListener('click', e=>{
  if (e.target.tagName==='IMG'){ document.getElementById('lbimg').src = e.target.src; lb.style.display='flex'; }
});
lb.onclick = ()=> lb.style.display='none';

const start = parseInt((location.hash.match(/^#f(\d+)$/)||[])[1]);
rebuild(isNaN(start) ? F[0].fid : start);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
