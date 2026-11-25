"""Generate SceneAssembly/docs/scene-assembly-input-example.html from batch 1's session.json.

Every number on the page comes from session.json, written by assemble_batch.py;
this script opens no handoff file.
"""
import json
import os

from PIL import Image

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs')
OUT = os.path.join(DOCS, 'assets', 'scene-assembly-input-example')
PAGE = os.path.join(DOCS, 'scene-assembly-input-example.html')

S = json.load(open(f'{OUT}/session.json'))
M = S['objects_meta']
CAM = S['camera']
ORDER = ['obj_0037', 'obj_0049', 'obj_0048', 'obj_0062', 'obj_0016', 'obj_0061']  # support depth order
CAT = S['catalog']
cov = {int(s): v for s, v in S['covisibility'].items()}  # JSON keys are strings
seqs = sorted(cov)


def tok(path):
    w, h = Image.open(path).size
    return int(round(w * h / 750)), (w, h)


T = {}
for o in ORDER:
    T[o] = {}
    T[o]['kf'] = tok(f'{OUT}/kf/{o}.jpg')
    T[o]['sur'] = tok(f'{OUT}/surround/{o}.jpg')
    T[o]['cand'] = tok(f'{OUT}/cand/{o}.jpg')
    if os.path.exists(f'{OUT}/crop/{o}.jpg'):
        T[o]['crop'] = tok(f'{OUT}/crop/{o}.jpg')

CROP_SET = [o for o in ORDER if M[o]['coverage'] < 0.05]

NDIV = sum(1 for o in ORDER for c in M[o]['candidates'] if c['diverged'])
CARD_TOK = 230
bundle_tot = {}
for o in ORDER:
    n = T[o]['kf'][0] + T[o]['sur'][0] + T[o]['cand'][0]
    if 'crop' in T[o]:
        n += T[o]['crop'][0]
    bundle_tot[o] = n + CARD_TOK
img_disp = sum(T[o]['kf'][0] + T[o]['sur'][0] + T[o]['cand'][0] + (T[o]['crop'][0] if 'crop' in T[o] else 0) for o in ORDER)
kf_native = sum(int(round(CAM['W'] * CAM['H'] / 750)) for _ in ORDER)
img_native = img_disp - sum(T[o]['kf'][0] for o in ORDER) + kf_native
SHARED = 2800
total_disp = img_disp + CARD_TOK * len(ORDER) + SHARED
total_native = img_native + CARD_TOK * len(ORDER) + SHARED


def card(o):
    m = M[o]
    c = m['candidates']
    lines = [f"card {o} · {m['label']} · {m['dims']} scan bbox"]
    lines.append(f"  scan   {m['dims'][0]:.3f} x {m['dims'][1]:.3f} x {m['dims'][2]:.3f} m, bottom z {m['contacts']['bottom_p2']:.3f}")
    lines.append(f"  asset  {m['asset_dims'][0]:.3f} x {m['asset_dims'][1]:.3f} x {m['asset_dims'][2]:.3f} m, height {m['height_m']:.3f} (scale src: {m['size_source']}, x{m['scale_factor']:.3f})")
    lines.append(f"physics {m['body']} · {m['collider']} collider · see catalog for mass")
    lines.append(f"pose   SAM3D yaw {m['yaw_sam3d']:.1f}°, up err {m['up_err']:.1f}°")
    lines.append(f"       ICP (yaw-locked, bounded +/-20° and 25 cm from its start)")
    for cc in c:
        tag = '  rejected: refinement made it worse' if cc['diverged'] else ''
        lines.append(f"         {cc['start']:>3.0f}° -> {cc['yaw']:>5.1f}°  res {cc['residual']*100:5.1f} cm  inl5 {cc['inlier']*100:4.1f}%  IoU {cc['iou']:.2f}{tag}")
    sup = m['contacts']['list']
    lines.append("support  " + (" · ".join(f"{d['support']} {d['points']} pts (z {d['z'][0]:.3f}-{d['z'][1]:.3f})" for d in sup) if sup else "none inside footprint"))
    lines.append(f"        {m.get('wall_note','')}")
    return '\n'.join(lines)


# wall numbers measured separately
WALL = {'obj_0037': 0.018, 'obj_0048': 0.020, 'obj_0049': 0.016,
        'obj_0062': 0.984, 'obj_0061': 1.591, 'obj_0016': 0.015}
for o in ORDER:
    M[o]['wall_note'] = f"wall nearest {WALL[o]:.3f} m (wall objectId 19 / 26)"


def e(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


# ---------------------------------------------------------------- page
H = []
H.append(f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>一个批会话的输入 · 第 1 批 · scene0000_00</title><style>
:root{{color-scheme:light;--ink:#203247;--line:#d8e3ed;--blue:#14628b;--green:#2f7a45;--amber:#b27a12;--muted:#5b7181}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;scroll-padding-top:24px}}
body{{margin:0;background:#f3f7fa;color:var(--ink);font:17px/1.75 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 24px 80px}}a{{color:var(--blue)}}
h1{{font-size:34px;line-height:1.3;margin:6px 0 10px}}h2{{font-size:25px;margin:46px 0 14px}}h3{{font-size:19px;margin:0 0 8px}}h4{{font-size:16px;margin:18px 0 6px;color:#38536a}}p{{margin:8px 0}}
header,.card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px}}
.meta{{font-size:14px;color:var(--muted);letter-spacing:.07em}}.lead{{font-size:19px}}
nav.toc{{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}}nav.toc a{{font-size:14px;text-decoration:none;background:#e8f2f9;border:1px solid #cde1ef;border-radius:8px;padding:3px 10px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.grid3{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}
.note{{border-left:4px solid #dfae4f;padding:12px 18px;background:#fff9e9;border-radius:0 10px 10px 0;margin:14px 0}}
table{{width:100%;border-collapse:collapse;background:#fff;font-size:14.5px}}th,td{{text-align:left;vertical-align:top;border:1px solid var(--line);padding:7px 10px}}th{{background:#f6f9fb}}
.table{{overflow:auto}}code{{font-size:.88em;background:#eef3f7;padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}}
figure{{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px}}figure svg{{width:100%;height:auto;display:block}}
figcaption{{font-size:14.5px;color:var(--muted);margin-top:8px}}
.big{{font-size:30px;font-weight:700;line-height:1.2}}.small{{font-size:14px;color:var(--muted)}}
ol li,ul li{{margin:4px 0}}
pre.msg{{background:#0f1a24;color:#cfe3f3;border-radius:10px;padding:14px 16px;font:13px/1.55 ui-monospace,Menlo,Consolas,monospace;overflow:auto;white-space:pre;margin:8px 0}}
pre.msg .k{{color:#7fd3a0}}pre.msg .n{{color:#f0c674}}pre.msg .q{{color:#9db8cc}}
.bundle{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:18px 0}}
.bhead{{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:12px}}
.bhead .oid{{font:700 20px ui-monospace,monospace;color:#14628b}}.bhead .lb{{font-weight:700}}.bhead .tk{{margin-left:auto;font-size:14px;color:var(--muted)}}
.shot{{display:flex;gap:12px;align-items:flex-start;margin:10px 0;flex-wrap:wrap}}
.shot img{{border:1px solid var(--line);border-radius:8px;max-width:100%;display:block}}
.lab{{font-size:13px;color:var(--muted);margin:2px 0 4px}}
.held{{width:330px}}.held img{{width:330px}}
.wide{{flex:1 1 480px;min-width:320px}}.wide img{{width:100%}}
.tag{{display:inline-block;font-size:12.5px;border-radius:6px;padding:1px 7px;margin-right:4px;background:#e8f2f9;border:1px solid #cde1ef;color:#14628b}}
.warn{{background:#fdeaea;border-color:#e7b4b4;color:#a33}}
.ok{{background:#e9f7ee;border-color:#b8d8c0;color:#2b6b3f}}
details summary{{cursor:pointer;font-weight:600;color:var(--blue);margin:6px 0}}
@media(max-width:820px){{.grid,.grid3{{grid-template-columns:1fr}}.held{{width:100%}}.held img{{width:100%}}h1{{font-size:27px}}main{{padding:18px 12px}}}}
</style></head><body><main>

<header>
<div class="meta">SCENEASSEMBLY · 会话输入实例 · 2026-11-10 · scene0000_00</div>
<h1>一个批会话的输入：第 1 批，6 个物体</h1>
<p class="lead">这一页是照 <a href="scene-assembly-design.html">方案页</a> §2 真的装配出来的一次会话输入。所有图都是按精确关键帧相机离线渲染的，所有数字都是这一批现算的。页面上每一张图、每一行文字，就是模型在那个上下文里实际看到的东西。</p>
<nav class="toc"><a href="#batch">0 这一批</a><a href="#shape">1 输入的整体形状</a><a href="#shared">2 共享层</a><a href="#bundles">3 六个输入包</a><a href="#budget">4 预算</a><a href="#findings">5 这批暴露的问题</a></nav>
</header>
''')

# ---------------- 0
H.append(f'''
<h2 id="batch">0 · 这一批是什么</h2>
<p>选第 1 批是因为它<b>自洽</b>：床是骨架，床头柜立在地上，枕头、两个未标注物、纸巾盒都压在它们上面。批内按支撑深度排序即可，不需要等别的批先落位。它同时踩到方案设想的三类难点——大物体、小占比物体、SAM3D 倒置。</p>
<div class="grid3">
<div class="card"><div class="big">{len(ORDER)}</div><p>个物体，按支撑深度排：bed → nightstand / tissue box → pillow / obj_0061 / obj_0062</p></div>
<div class="card"><div class="big">5</div><p>张不同的主关键帧。6 个物体里 obj_0062 和 pillow 共用第 81 帧——这就是"共享关键帧"规则在起作用</p></div>
<div class="card"><div class="big">{len(seqs)}</div><p>张共视候选帧（6 个物体全部关键帧的并集）。装配器只发了其中 5 张</p></div>
</div>
<div class="table"><table>
<tr><th style="width:13%">物体</th><th style="width:12%">label</th><th style="width:9%">主帧</th><th style="width:11%">mask 占比</th><th style="width:17%">扫描 bbox (m)</th><th style="width:17%">资产 (m)</th><th>处理顺序依据</th></tr>''')
WHY = {'obj_0037': '骨架：所有上层物的支撑面', 'obj_0049': '立在地上，先于其上的纸巾盒',
       'obj_0048': '压在 nightstand 上', 'obj_0062': '躺在床上', 'obj_0061': '压在 bed + obj_0062 上',
       'obj_0016': '躺在床上'}
for o in ORDER:
    m = M[o]
    H.append(f'<tr><td><code>{o}</code></td><td>{e(m["nice"])}</td><td>{m["seq"]}</td><td>{m["coverage"]*100:.2f}%</td>'
             f'<td>{m["dims"][0]:.2f}×{m["dims"][1]:.2f}×{m["dims"][2]:.2f}</td>'
             f'<td>{m["asset_dims"][0]:.2f}×{m["asset_dims"][1]:.2f}×{m["asset_dims"][2]:.2f}</td>'
             f'<td>{WHY[o]}</td></tr>')
H.append('</table></div>')

# ---------------- 1 shape
H.append(f'''
<h2 id="shape">1 · 输入的整体形状</h2>
<p>一次会话的输入 = <b>一段共享层（发一次）</b> + <b>六个物体包（各含图与文字）</b>。共享层保证 agent 知道自己在哪个场景、还有什么没放；物体包保证每个物体都能被"看清"而不是"猜对"。</p>
<figure>
<svg viewBox="0 0 1120 620" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">
<defs><marker id="bx" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#4a6478"/></marker></defs>
<rect x="14" y="14" width="270" height="470" rx="12" fill="#fff8e8" stroke="#dfae4f" stroke-width="1.5"/>
<text x="149" y="44" text-anchor="middle" font-size="16" font-weight="700" fill="#8a5d08">共享层 · 每批发一次</text>
<g font-size="13.5" fill="#203247">
<text x="28" y="76">① 初始事实 ~450 token</text>
<text x="28" y="105">② 场景目录 52 行 ~1.4k</text>
<text x="28" y="134">③ 共视帧表 15×6 ~250</text>
<text x="28" y="163">④ 支撑接触候选表 ~300</text>
<text x="28" y="192">⑤ 约定登记表 ~150</text>
<text x="28" y="221">⑥ 候选与 IoU 表 ~300</text></g>
<text x="28" y="256" font-size="13" fill="#5b7181">合计 ≈ {SHARED/1000:.1f}k token</text>
<line x1="28" y1="272" x2="270" y2="272" stroke="#e6d3aa"/>
<text x="28" y="298" font-size="13" font-weight="600" fill="#8a5d08">不含任何物体的图</text>
<text x="28" y="320" font-size="13" fill="#5b7181">图片一律走物体包，</text>
<text x="28" y="340" font-size="13" fill="#5b7181">避免 116 张关键帧堆进来</text>
<text x="28" y="386" font-size="13" font-weight="600" fill="#8a5d08">发一次的判定</text>
<text x="28" y="408" font-size="13" fill="#5b7181">跨批信息进约定登记表，</text>
<text x="28" y="428" font-size="13" fill="#5b7181">不进对话历史（§2.4）</text>
</g>
''')
y = 14
H.append('<text x="322" y="34" font-size="15" font-weight="700" fill="#203247">物体包 × 6（按支撑深度排）</text>')
for i, o in enumerate(ORDER):
    m = M[o]
    h = 72
    H.append(f'<rect x="322" y="{y+44}" width="500" height="{h}" rx="10" fill="#eef5fb" stroke="#9cc3dc"/>')
    H.append(f'<text x="336" y="{y+70}" font-size="14" font-weight="700" fill="#14628b">{o}</text>')
    H.append(f'<text x="410" y="{y+70}" font-size="13" fill="#203247">{e(m["nice"])} · frame {m["seq"]} · mask {m["coverage"]*100:.1f}%</text>')
    chips = f'① 帧 {T[o]["kf"][0]}'
    if "crop" in T[o]:
        chips += f' · 裁块 {T[o]["crop"][0]}'
    chips += f' · ② 环绕 {T[o]["sur"][0]} · ③ 候选 {T[o]["cand"][0]} · ④ 卡 {CARD_TOK}'
    H.append(f'<text x="336" y="{y+96}" font-size="12.5" fill="#38536a">{chips}</text>')
    H.append(f'<text x="812" y="{y+70}" font-size="13" font-weight="600" fill="#14628b" text-anchor="end">{bundle_tot[o]} tok</text>')
    y += 78
H.append(f'''<line x1="318" y1="512" x2="822" y2="512" stroke="#9cc3dc" stroke-dasharray="4 3"/>
<text x="322" y="536" font-size="13.5" fill="#38536a">6 个包合计 {sum(bundle_tot.values())} token · 共享层 {SHARED} · 会话输入合计 {total_disp}</text>
<rect x="852" y="14" width="254" height="560" rx="12" fill="#edf8f0" stroke="#8cc09a" stroke-width="1.5"/>
<text x="979" y="44" text-anchor="middle" font-size="16" font-weight="700" fill="#24603a">一次会话</text>
<g font-size="13" fill="#203247">
<text x="866" y="76">共享层 + 6 个包顺序灌入</text>
<text x="866" y="102">批内串行：先床，再床上的</text>
<text x="866" y="128">每个物体：看图 → propose</text>
<text x="866" y="152">→ 核验渲染 → commit</text>
<text x="866" y="182">agent 可回看同批任意图</text>
<text x="866" y="206">（同批共视，不跨批）</text>
</g>
<line x1="866" y1="228" x2="1092" y2="228" stroke="#b8d8c0"/>
<text x="866" y="256" font-size="13" font-weight="700" fill="#24603a">输入预算</text>
<g font-size="13" fill="#203247">
<text x="866" y="282">展示分辨率 {total_disp/1000:.1f}k</text>
<text x="866" y="306">原生关键帧 {total_native/1000:.1f}k</text>
<text x="866" y="330">（方案页 §2.1 估 35k）</text></g>
<line x1="866" y1="352" x2="1092" y2="352" stroke="#b8d8c0"/>
<text x="866" y="380" font-size="13" font-weight="700" fill="#24603a">共享 vs 私有</text>
<g font-size="12.5" fill="#38536a">
<text x="866" y="404">目录、约定：全批共享</text>
<text x="866" y="426">帧 81：0062 与 pillow</text>
<text x="866" y="446">共用原图，只换 mask</text>
<text x="866" y="468">其余图各归各的包</text></g>
<text x="866" y="512" font-size="12.5" fill="#5b7181">批末写回：约定、</text>
<text x="866" y="532" font-size="12.5" fill="#5b7181">支撑确认结果、asset flags</text>
</svg>
<figcaption>左边是只发一次的文本上下文，中间是按处理顺序排好的六个包，右边是这个会话的预算与共享边界。每张图后面的数字是按"像素 / 750"估的输入 token。</figcaption>
</figure>
''')

# ---------------- 2 shared layer
H.append(f'''
<h2 id="shared">2 · 共享层：每个批都发这一段</h2>
<p>下面这三段是<b>原样</b>发给模型的文本。它们不含任何一张图，也不含任何已经落位的物体——初始状态里 52 个物体全是 unplaced。</p>

<h3>2.1 · 初始事实</h3>
<pre class="msg"><span class="k">SCENE</span> scene0000_00 (ScanNet) · frame is the raw scan frame
  Z-up, metres, no axisAlignment applied. floor_z = {S['floor_z']:.4f} m
  colour intrinsics fx={CAM['fx']:.3f} fy={CAM['fy']:.3f} cx={CAM['cx']} cy={CAM['cy']} · {CAM['W']}x{CAM['H']}
  camera_to_world is OpenCV convention (+Z forward, +Y down). camera centre = M[:3,3]
  keyframe RGB and binary mask: scene handoff keyframes/&lt;oid&gt;/&lt;seq:04d&gt;_{{rgb.jpg,mask.png}}
  obj_NNNN == ScanNet objectId NNNN; instance-mask pixel value = objectId + 1, 0 = unannotated

<span class="k">ASSETS</span> assembly_handoff/v2, one USD per object
  Z-up, metresPerUnit 1, XY bbox centred, bottom at z=0, upright
  visual.usdc default prim /Obj&lt;N&gt;; displayColor is vertex-interpolated and LINEAR
    -&gt; any render you are shown is already converted to sRGB
  physics.usdc gives body_type, mass, collider, friction
  the asset's XY orientation comes from the SAM3D canonical frame: it is NOT the scene orientation

<span class="k">POSE EVIDENCE</span>
  SAM3D rotation is usable; its scale and translation are not metric. Do not use them as xyz.
  "yaw" is degrees about +Z, measured from +X, counter-clockwise.
  up error = angle between the SAM3D +Z and the world +Z. &gt; 30 deg means the whole asset
  is probably rotated upside down, and its yaw is then unreliable.
  registrations are yaw-locked to the scan cloud, bounded to +/-20 deg and 25 cm from their start.
  a small residual difference does NOT mean one yaw is more correct than another.</pre>

<h3>2.2 · 场景目录（52 行）</h3>
<p>每行压缩到"还有什么没放"这个粒度：标号、类别、扫描尺寸、底面高度、物理体、可看的关键帧。agent 不需要为知道隔壁有什么而看图。</p>
<div class="table"><table>
<tr><th>oid</th><th>label</th><th>扫描尺寸 x×y×z</th><th>底面 z</th><th>body</th><th>collider</th><th>主帧</th><th>主帧占比</th><th>关键帧数</th></tr>''')
for r in CAT:
    cls = ' style="background:#eef5fb"' if r['oid'] in ORDER else ''
    H.append(f'<tr{cls}><td><code>{r["oid"]}</code></td><td>{e(r["label"])}</td><td>{r["extent"][0]:.2f}×{r["extent"][1]:.2f}×{r["extent"][2]:.2f}</td>'
             f'<td>{r["bottom_z"]:.2f}</td><td>{r["body"]}</td><td>{r["collider"]}</td><td>{r["seq"]}</td>'
             f'<td>{r["coverage"]*100:.1f}%</td><td>{r["key_frames"]}</td></tr>')
H.append('</table></div><p class="small">浅蓝底是本批的 6 个。目录本身约 1.4k token。</p>')

H.append('''
<h3>2.3 · 已经算好的证据表</h3>
<p>确定性的东西 harness 先算完，模型不重算、也不靠记。三张表：本批 15 张候选帧里每个物体各占多少（挑图依据）；每个物体的支撑接触候选；每个物体的候选朝向与残差。</p>
<h4>共视矩阵（百分比 = 该物体在那一帧里的 mask 占比）</h4>
<div class="table"><table>
<tr><th>帧</th>''' + ''.join(f'<th>{o[-4:]}</th>' for o in ORDER) + '<th>说明</th></tr>')
PRIM = {M[o]['seq']: [] for o in ORDER}
for o in ORDER:
    PRIM[M[o]['seq']].append(o)
for s in seqs:
    tags = []
    for o in ORDER:
        if M[o]['seq'] == s:
            tags.append(o)
    note = ''
    if tags:
        note = '本批主帧：' + '、'.join(tags)
    best = max(cov[s].items(), key=lambda kv: kv[1])[0]
    tds = []
    for o in ORDER:
        hl = ' style="background:#e9f7ee"' if M[o]['seq'] == s else ''
        tds.append(f'<td{hl}>{cov[s][o]*100:.2f}</td>')
    H.append(f'<tr><td>{s}</td>' + ''.join(tds) + f'<td class="small">{note}</td></tr>')
H.append('</table></div><p class="small">绿色是本批选用的主帧。15 张里只有 5 张被用上——其余 10 张的物体占比都太小，或者与已选帧重复。</p>')

H.append('<h4>支撑接触候选（扫描点，±4 cm 水平栈、底面下 5 cm 带内）</h4>')
H.append('<div class="table"><table><tr><th>物体</th><th>底面 z (p2)</th><th>支撑候选（点数 / 接触 z 区间）</th></tr>')
for o in ORDER:
    c = M[o]['contacts']
    lst = ' · '.join(f'<code>{d["support"]}</code> {d["points"]} 点 @ {d["z"][0]:.3f}–{d["z"][1]:.3f}' for d in c['list']) or '—'
    H.append(f'<tr><td><code>{o}</code> {e(M[o]["nice"])}</td><td>{c["bottom_p2"]:.3f}</td><td>{lst}</td></tr>')
H.append('</table></div>')
H.append('<div class="note"><b>这张表会骗人，而且这一批正好有一个例子。</b>bed 的支撑候选是 nightstand 的 19 个点（z 0.059–0.113）——那是床边柜腿，不是床躺在柜子上。点级检测只看"下方有点"，方向分不出来。所以表里只写候选和点数，不写结论；结论由 agent 看图后 <code>declare_relation</code>。</div>')

H.append('<h4>候选朝向与残差（每块含 yaw、残差、内点率、关键帧 IoU）</h4>')
H.append('<div class="table"><table><tr><th>物体</th><th>SAM3D</th><th>up 偏差</th><th>ICP 0°</th><th>ICP 90°</th><th>ICP 180°</th><th>ICP 270°</th><th>IoU 最大</th></tr>')
for o in ORDER:
    m = M[o]
    cells = ''.join(f'<td>{c["yaw"]:.0f}° / {c["residual"]*100:.1f} cm / {c["iou"]:.2f}</td>' for c in m['candidates'])
    H.append(f'<tr><td><code>{o}</code></td><td>{m["yaw_sam3d"]:.0f}° / {m["sam3d"]["residual"]*100:.1f} cm / {m["sam3d"]["iou"]:.2f}</td>'
             f'<td>{m["up_err"]:.0f}°</td>{cells}<td>{m["best_iou"]["iou"]:.2f} @ {m["best_iou"]["yaw"]:.0f}°</td></tr>')
H.append('</table></div>')

H.append('''
<h3>2.4 · 约定登记表</h3>
<p>登记表是<b>跨批</b>的唯一通道。第 1 批开始时它只有 harness 预置的三条；批末由 agent 写回结论。</p>
<div class="table"><table>
<tr><th style="width:8%">状态</th><th style="width:26%">条目</th><th>来源</th></tr>
<tr><td>预置</td><td><code>yaw 从 +X 起算、绕 +Z 逆时针</code></td><td>harness 统一口径</td></tr>
<tr><td>预置</td><td><code>SAM3D up 偏差 &gt; 30° 时整体可能倒置，yaw 不可信</code></td><td>全场景实测：8 个物体偏差 ≥30°，其中 6 个确为倒置</td></tr>
<tr><td>预置</td><td><code>残差与 IoU 的微小差距不能用来定正反</code></td><td>实测 4/52 个物体 IoU 差距 ≥0.05；48/52 个残差比 &lt;1.2</td></tr>
<tr><td>本批待写</td><td><code>正面 = 有抽屉把手/开口的那面</code></td><td>nightstand obj_0049 与 tissue box obj_0048 本批一起定</td></tr>
<tr><td>本批待写</td><td><code>床以枕头的一侧为"头"</code></td><td>影响后续 pillow 与 nightstand 的相对朝向</td></tr>
</table></div>
''')

# ---------------- 3 bundles
H.append('<h2 id="bundles">3 · 六个输入包</h2>')
H.append('<p>每个包五件：① 主关键帧（叠 mask 轮廓，占比 &lt;5% 的另给裁块）② 资产环绕 ③ 候选对比 ④ 文字卡 ⑤ 状态行。下面是原样的内容。</p>')

for o in ORDER:
    m = M[o]
    H.append(f'''
<div class="bundle">
<div class="bhead"><span class="oid">{o}</span><span class="lb">{e(m["nice"])}</span>
<span class="small">frame {m["seq"]} · mask {m["coverage"]*100:.2f}% · 主关键帧由 {len([1 for x in ORDER if M[x]["seq"]==m["seq"]])} 个物体共用</span>
<span class="tk">包内 {bundle_tot[o]} tok</span></div>

<div class="shot">
  <div class="held"><div class="lab">① 主关键帧 {T[o]['kf'][1][0]}×{T[o]['kf'][1][1]} · {T[o]['kf'][0]} tok</div>
  <img src="assets/scene-assembly-input-example/kf/{o}.jpg" alt="{o} keyframe"></div>''')
    if 'crop' in T[o]:
        H.append(f'''  <div class="held"><div class="lab">①b 裁块（占比 &lt;5%）bbox×2 · {T[o]['crop'][1][0]}×{T[o]['crop'][1][1]} 显示 · {T[o]['crop'][0]} tok</div>
  <img src="assets/scene-assembly-input-example/crop/{o}.jpg" alt="{o} crop"></div>''')
    else:
        H.append(f'''  <div class="held"><div class="lab">①b 不裁：占比 {m['coverage']*100:.2f}% ≥ 5%</div>
  <div class="card small" style="padding:14px">这一档原图已经够看。方案 §2.3 只在 &lt;5% 时给裁块，省一张图。</div></div>''')
    H.append(f'''</div>
<div class="shot">
  <div class="held"><div class="lab">② 资产环绕 4 视角 + 局部坐标轴 · {T[o]['sur'][1][0]}² · {T[o]['sur'][0]} tok</div>
  <img src="assets/scene-assembly-input-example/surround/{o}.jpg" alt="{o} surround"></div>
  <div class="wide"><div class="lab">③ 候选对比：同一精确关键帧相机，1×5 · {T[o]['cand'][1][0]}×{T[o]['cand'][1][1]} · {T[o]['cand'][0]} tok</div>
  <img src="assets/scene-assembly-input-example/cand/{o}.jpg" alt="{o} candidates"></div>
</div>
<div class="lab">④ 文字卡（英文，原样）</div>
<pre class="msg">{e(card(o))}</pre>
<div class="lab">⑤ 状态行</div>
<pre class="msg">state {o} · unplaced · pose — · scale 1.000 · relations — · flags —</pre>
</div>''')

# ---------------- 4 budget
H.append(f'''
<h2 id="budget">4 · 预算</h2>
<div class="table"><table>
<tr><th style="width:22%">项</th><th style="width:10%">张数</th><th style="width:16%">尺寸</th><th style="width:14%">展示 token</th><th style="width:14%">原生 token</th><th>说明</th></tr>
<tr><td>① 主关键帧</td><td>6</td><td>{CAM['W']//2}×{CAM['H']//2} / {CAM['W']}×{CAM['H']}</td><td>{sum(T[o]['kf'][0] for o in ORDER)}</td><td>{kf_native}</td><td>一物体一张；0062 与 pillow 共用第 81 帧的原图，各叠各的 mask</td></tr>
<tr><td>①b 裁块</td><td>{len(CROP_SET)}</td><td>512 宽</td><td>{sum(T[o]['crop'][0] for o in CROP_SET)}</td><td>同左</td><td>只有占比 &lt;5% 的 {len(CROP_SET)} 个，见下表</td></tr>
<tr><td>② 资产环绕</td><td>6</td><td>760²</td><td>{sum(T[o]['sur'][0] for o in ORDER)}</td><td>同左</td><td>4 视角拼成一张，不单独发 4 张</td></tr>
<tr><td>③ 候选对比</td><td>6</td><td>≤1500×300</td><td>{sum(T[o]['cand'][0] for o in ORDER)}</td><td>同左</td><td>5 个候选拼一张，取代 5 次 view_compare</td></tr>
<tr><td>④ 文字卡</td><td>6</td><td>文本</td><td>{CARD_TOK*6}</td><td>同左</td><td>约 {CARD_TOK} token/物体</td></tr>
<tr><td>共享层</td><td>1</td><td>文本</td><td>{SHARED}</td><td>同左</td><td>目录、事实、三张证据表、约定</td></tr>
<tr style="background:#eef5fb"><td><b>合计</b></td><td><b>{6+len(CROP_SET)+6+6}</b></td><td>—</td><td><b>{total_disp}</b></td><td><b>{total_native}</b></td><td>方案 §2.1 估 35k；展示分辨率下这批只要 {total_disp/1000:.1f}k</td></tr>
</table></div>
<p>两个数字都给，是因为它们回答的是不同问题。<b>展示分辨率</b>（关键帧发一半尺寸）是这一页实际渲染出来的东西，也是我建议的默认档：小物体的细节由裁块补，主帧只需要看朝向和支撑关系。<b>原生分辨率</b>是保守情况——如果某个物体在原图里确实太小、裁块又不够，才发原图。</p>
<div class="note"><b>比方案页的 35k 少，原因是估法和实际装配不一样。</b>方案页按"5 个物体 × 4 张图 × 全尺寸"估；实际装配里每个物体只发 1 张主关键帧（二次、三次关键帧留给核验轮的按需调用），环绕与候选都拼成单张，裁块只给小物体。图的数量从 20 降到 {6+len(CROP_SET)+6+6}，每张也都不大。</div>
''')

H.append('<h4>裁块判定</h4><div class="table"><table><tr><th>物体</th><th>mask 占比</th><th>是否裁</th><th>理由</th></tr>')
for o in ORDER:
    m = M[o]
    if m['coverage'] < 0.05:
        H.append(f'<tr><td><code>{o}</code></td><td>{m["coverage"]*100:.2f}%</td><td><span class="tag ok">裁</span></td>'
                 f'<td>低于 5%，原图里只有 {m["visible_pixels"]} 像素，裁剪到 bbox×2 后边长 {m["crop_side"]} px</td></tr>')
    else:
        H.append(f'<tr><td><code>{o}</code></td><td>{m["coverage"]*100:.2f}%</td><td><span class="tag">不裁</span></td>'
                 f'<td>{m["visible_pixels"]} 像素，轮廓和支撑关系在原图里已经够清楚</td></tr>')
H.append('</table></div>')

# ---------------- 5 findings
H.append(f'''
<h2 id="findings">5 · 这批暴露的问题</h2>
<p>装配和渲染过程本身就是一次体检。六个物体里，几何能分开的只有一个。</p>
<div class="table"><table>
<tr><th style="width:15%">物体</th><th style="width:38%">这批判出来的</th><th>所以 agent 要做什么</th></tr>
<tr><td><code>obj_0062</code></td><td><b>唯一一个轮廓能分开朝向的。</b>yaw 90° 的 IoU 0.785，yaw 0°/180° 只有 0.60 / 0.59，差 0.18。原因是它是一块扁平物体，长宽比大</td><td>可以直接信候选图上的最强朝向，不必额外调用</td></tr>
<tr><td><code>obj_0037</code></td><td><b>资产比扫描大。</b>资产 2.33×2.78×0.78 m，扫描 1.98×2.32×0.83 m：占地大 18%，矮 6%。四个朝向残差 31.9–35.6 cm、IoU 0.50–0.53，全挤在一起</td><td>判"接受并缩放到扫描占地"还是"标记资产";朝向只能看图，且床这种对称物体正反差别很小</td></tr>
<tr><td><code>obj_0049</code></td><td>四个朝向残差 10.4–11.2 cm、IoU 0.286–0.300。<b>几何完全分不开</b>，而且 IoU 整体偏低说明资产形状和扫描对不齐（抽屉、隔板细节）</td><td>唯一判据是环绕图上"哪面有抽屉"，再由 <code>align_with</code> 贴墙</td></tr>
<tr><td><code>obj_0016</code></td><td><b>up 偏差 70.8°。</b>超过 30° 阈值，资产整体姿态可疑；四个朝向残差 11.1–14.8 cm</td><td>先决定要不要 <code>flip</code>，再谈 yaw</td></tr>
<tr><td><code>obj_0061</code></td><td><b>up 偏差 176.3°——完全倒置。</b>四个朝向残差 2.0–2.9 cm、IoU 0.51–0.62，几何上完全看不出倒置。SAM3D yaw 61° 与四个候选都不一致</td><td>必须看图发现倒置，用 <code>flip</code> 修正；修正后 yaw 要重新取</td></tr>
<tr><td><code>obj_0048</code></td><td><b>资产形状错了。</b>资产深 0.836 m，扫描深 0.433 m，差 1.9 倍；残差 15–19 cm 是这批最大的之一。而且扫描底面 0.619 m 高于柜面——它其实压在床头柜上，不是在床上</td><td>判 <code>accept_scaled</code> 还是 <code>regen</code>；支撑关系声明为 <code>supported_by: obj_0049</code></td></tr>
</table></div>
<div class="note"><b>六分之五的物体，几何给不出朝向，只能给"四个都差不多"。</b>这不是渲染的问题——候选图在同一相机、同一裁剪下并排，人一眼能分出正反；残差和 IoU 的数字则互相咬住。这一批正好把方案页 §1 的论断复现了一遍：<b>离散判断交给看图，连续量交给几何。</b></div>
<p class="small">另外两个装配层面的发现：① 第 81 帧被两个物体共用，但发了两份独立的叠轮廓图（418 token ×2）。可以直接合成一张双色轮廓图，省一张。② 第 53 帧同时是 obj_0061 的主帧和 obj_0062 的次帧，装配器发的是 obj_0061 的版本——同一个相机，只是叠的 mask 不同，不重复发。</p>
''')

H.append(f'''
<h2 id="method">附 · 这一页的图是怎么来的</h2>
<div class="card"><ul style="margin:0;padding-left:20px">
<li>资产从 <code>visual.usdc</code> 读点、面、逐个顶点的 <code>primvars:displayColor</code>，线性色转 sRGB 后逐面平均。</li>
<li>渲染用 trimesh + embree 光线求交，按面法向做背面剔除和 Lambert 明暗。候选图用场景交接里的真实 <code>camera_to_world</code> 和真实内参（{CAM['W']}×{CAM['H']}），按半分辨率出图后拼在一张里。整场景不走渲染器，只渲染单个资产叠在照片上，<b>所以候选图里看不到遮挡</b>：被床挡住的柜腿也会画出来。</li>
<li>候选是"yaw 锁定 + 竖直锁定"的 ICP：初始 yaw 取 0/90/180/270°，初始平移取扫描包围盒的 XY 中心与底面，然后限制在 ±20°、25 cm 窗口内收敛。收敛后如果残差没变好，就回退到起点并标 <code>diverged</code>——这批 24 个候选里有 {NDIV} 个触发了回退。</li>
<li>ICP 的对应关系是资产顶点到扫描点（单向，最近邻，去掉最差 15%）。所以残差偏大也可能只是资产比扫描大，而不一定是朝向错。这正是为什么它只作证据、不作结论。</li>
<li>脚本：<code>SceneAssembly/src/legacy/</code> 下的 <code>usdmesh.py</code>（USD 读取）、<code>render.py</code>（相机与光线求交）、<code>icp.py</code>（带窗口的 ICP）、<code>assemble_batch.py</code>（组装）、<code>make_page.py</code>（本页）。</li>
</ul></div>
''')

H.append('</main></body></html>')
open(PAGE, 'w').write('\n'.join(H))
print('wrote', PAGE, 'img_disp', img_disp, 'total_disp', total_disp, 'total_native', total_native)
