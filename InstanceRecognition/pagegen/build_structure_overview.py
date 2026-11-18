"""The structure-recognition overview page: the whole chain on one page, every number from the artifacts.

    python -m pagegen.build_structure_overview --run struct_v4_win5     # -> docs/structure-overview.html

One section per step (2D segmentation, the delivery frame, lift, rays, wall planes,
closure, handoff): what goes in, what is done, what comes out on scene0000_00, which
values are measured, assumed or tuned. Then the chain against ScanNet GT and a list of
weak points to work from.

Reads src.structure's outputs (artifacts/structure/<run>/), the frame file, the wall
handoff, and - for the GT columns only - the ScanNet 2D instance annotation and the GT
structure page data (docs/assets/structure-gt/data.js). Nothing here feeds back.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.geometry.floor_frame import frame_path
from src.structure.frames import OUT_ROOT, RAYS, masks_dir

from .build_structure_geom import GT_DATA, gt_reference, match_gt

DOCS = config.ROOT / "docs"
OUT = DOCS / "structure-overview.html"
ASSETS = DOCS / "assets/structure-overview"
SEG_PAGE_ASSETS = "assets/structure-{run}"          # per-frame rgb/kept/gt pictures of the segmentation page
HANDOFF = config.ARTIFACTS_ROOT / "handoff/scene0000_00/walls_harness"
SCENE = config.SCANNET_ROOT / "scene0000_00"
LABELS = ("wall", "floor", "door", "window")
ZH = {"wall": "墙", "floor": "地面", "door": "门扇", "window": "窗"}
# what counts as right inside a mask of each label (doorframe trim inside a wall or door mask is fine)
RIGHT = {"wall": {"wall", "doorframe"}, "floor": {"floor"}, "door": {"door", "doorframe"}, "window": {"window"}}
EXAMPLE_FRAMES = (1229, 3800, 2794)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

esc = html.escape


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{x * 100:.0f}%"


# ---------------------------------------------------------------- data
def segmentation(run: str) -> dict:
    seg = masks_dir(run)
    cfg = json.loads((seg / "config.json").read_text())
    wins = [json.loads(p.read_text()) for p in sorted(seg.glob("window_*/window.json"))]
    frames = [json.loads(p.read_text()) for p in sorted(seg.glob("frame_*/result.json"))]
    kept = Counter(k["label"] for r in frames for k in r["kept"])
    return {"cfg": cfg, "windows": len(wins), "rounds": [w["rounds"] for w in wins],
            "prompt_tokens": sum(w["usage"]["prompt_tokens"] for w in wins),
            "completion_tokens": sum(w["usage"]["completion_tokens"] for w in wins),
            "seconds": sum(w["seconds"] for w in wins), "frames": len(frames), "kept": kept,
            "not_planned": [n for w in wins for n in w["not_planned_notes"]],
            "segment_calls": Counter(r["segment_calls"] for r in frames), "results": frames, "dir": seg}


def gt_purity(seg: dict) -> dict:
    """Per kept mask: share of its pixels on GT objects of its own class (doorframe allowed in wall/door)."""
    agg = json.loads((SCENE / "scene0000_00.aggregation.json").read_text())
    labels = {int(g["objectId"]): g["label"] for g in agg["segGroups"]}
    z = zipfile.ZipFile(SCENE / "scene0000_00_2d-instance.zip")
    rows, missed_floor, frames_with = [], [], Counter()
    for r in seg["results"]:
        fid = r["frame_id"]
        g = np.asarray(Image.open(io.BytesIO(z.read(f"instance/{fid}.png")))).astype(int) - 1
        share_gt = {c: float(np.isin(g, [i for i, x in labels.items() if x == c]).mean()) for c in LABELS}
        for c in LABELS:
            if share_gt[c] >= (0.05 if c == "floor" else 0.01):
                frames_with[c] += 1
        for k in r["kept"]:
            m = np.asarray(Image.open(seg["dir"] / f"frame_{fid:06d}" / k["mask_file"]).convert("L")) > 0
            v = g[m]; v = v[v >= 0]
            by = Counter(labels.get(int(i), "?") for i in v)
            tot = max(1, sum(by.values()))
            purity = sum(n for c, n in by.items() if c in RIGHT[k["label"]]) / tot
            worst = max(((c, n / tot) for c, n in by.items() if c not in RIGHT[k["label"]]), key=lambda x: x[1], default=(None, 0))
            rows.append({"fid": fid, "label": k["label"], "pid": k["plan_id"], "where": k["where"], "purity": purity,
                         "area": k["area_fraction"], "worst": worst})
        if share_gt["floor"] >= 0.05 and not any(k["label"] == "floor" for k in r["kept"]):
            missed_floor.append((fid, share_gt["floor"]))
    per = {}
    for c in LABELS:
        rs = [x for x in rows if x["label"] == c]
        if rs:
            a = np.array([x["area"] for x in rs]); p = np.array([x["purity"] for x in rs])
            per[c] = {"n": len(rs), "mean": float(p.mean()), "area_weighted": float((p * a).sum() / a.sum()),
                      "frames_kept": len({x["fid"] for x in rs}), "frames_gt": frames_with[c]}
    bad = sorted([x for x in rows if x["purity"] < 0.75], key=lambda x: x["purity"])
    return {"per": per, "bad": bad, "missed_floor": missed_floor}


def gt_items() -> dict:
    s = GT_DATA.read_text()
    gt = json.loads(s[s.index("{"):s.index(";\nwindow.STRUCT.pos")])
    return {"faces": {f["key"]: f for f in gt["faces"]}, "items": gt["items"], "gaps": gt.get("gaps", [])}


def coverage(run: str, faces: list) -> list:
    """Per GT wall face: share of its 10 cm cells (0.1-2.3 m high) with any ray surface, with a wall-mask point, with a kept plane."""
    d = OUT_ROOT / run
    g = np.load(d / "geom.npz")
    X = g["X"][g["wall_first"]].astype(np.float64); plane = g["wall_plane"]
    P = np.load(RAYS)["P"].astype(np.float64)
    nk = len(json.loads((d / "walls.json").read_text())["walls"])
    out = []
    for f in faces:
        c, n, u = (np.asarray(f[k]) for k in ("c", "n", "u"))
        L = f["a"][1] - f["a"][0]
        tot = int(round(L / 0.1)) * 22

        def cells(Q):
            dd = (Q[:, :2] - c) @ n; a = (Q[:, :2] - c) @ u
            s = (np.abs(dd) < 0.2) & (a > f["a"][0]) & (a < f["a"][1]) & (Q[:, 2] > 0.1) & (Q[:, 2] < 2.3)
            return len(set(zip(np.floor(a[s] / 0.1).astype(int), np.floor(Q[s, 2] / 0.1).astype(int))))
        out.append({"key": f["key"], "name": f["name"], "length_m": L, "rays": min(1.0, cells(P) / tot),
                    "masks": min(1.0, cells(X) / tot), "kept": min(1.0, cells(X[(plane >= 0) & (plane < nk)]) / tot)})
    return out


def wall_tops(W: dict, gtf: dict, matches: dict) -> list:
    """Per kept wall: the highest ray surface on its face (p99 of points within 12 cm, above 1.8 m) vs the GT face top."""
    P = np.load(RAYS)["P"].astype(np.float64)
    out = []
    for r in W["walls"]:
        c, n, u = (np.asarray(r[k]) for k in ("point", "normal", "along"))
        dd = (P[:, :2] - c) @ n; a = (P[:, :2] - c) @ u
        s = (np.abs(dd) < 0.12) & (a > r["a"][0] + 0.1) & (a < r["a"][1] - 0.1) & (P[:, 2] > 1.8)
        m = matches.get(r["id"])
        out.append({"id": r["id"], "seen_top": float(np.percentile(P[s, 2], 99)) if s.sum() > 20 else None,
                    "gt_top": gtf[m["face"]]["z"][1] if m else None, "face": m["face"] if m else None})
    return out


def collect(run: str) -> dict:
    d = OUT_ROOT / run
    W = json.loads((d / "walls.json").read_text())
    C = json.loads((d / "closed.json").read_text())
    H = json.loads((HANDOFF / "walls.json").read_text())
    F = json.loads(frame_path().read_text())
    pts = np.load(d / "points.npz")
    rays = np.load(RAYS)
    seg = segmentation(run)
    faces, _ = gt_reference()
    gti = gt_items()
    matches = {r["id"]: match_gt(r, faces) for r in W["walls"] + W["rejected"]}
    # rectangle sides against the GT faces their walls lie on
    axes = [np.asarray(s["normal"]) for s in C["sides"]]
    rect_vs_gt = []
    fmap = {f["key"]: f for f in faces}
    for sd in C["sides"]:
        m = matches.get(sd["walls"][0]) if sd["walls"] else None
        if m:
            fc = np.asarray(fmap[m["face"]]["c"])
            rect_vs_gt.append({"side": sd["side"], "face": m["face"], "name": m["name"], "from": sd["from"],
                               "inward_cm": (sd["offset"] - fc @ axes[sd["k"]]) * 100})
    # room size per axis, against the distance between the GT faces the two opposite sides lie on
    by_side = {x["side"]: x for x in rect_vs_gt}
    size_vs_gt = []
    for k, (a, b, ours) in enumerate((("W", "E", C["rectangle"]["width_m"]), ("S", "N", C["rectangle"]["depth_m"]))):
        if a in by_side and b in by_side:
            ca, cb = (np.asarray(fmap[by_side[x]["face"]]["c"]) for x in (a, b))
            size_vs_gt.append({"axis": f"{a}–{b}", "ours": ours, "gt": abs((ca - cb) @ axes[k]),
                               "faces": f"{by_side[a]['face']} {by_side[a]['name']} ↔ {by_side[b]['face']} {by_side[b]['name']}"})
    return {"run": run, "W": W, "C": C, "H": H, "F": F, "seg": seg, "purity": gt_purity(seg), "size_vs_gt": size_vs_gt,
            "lift": {LABELS[i]: int((pts["label"] == i).sum()) for i in range(4)}, "points": int(len(pts["xyz"])),
            "rays": int(len(rays["P"])), "ray_frames": int(len(rays["cams"])), "matches": matches,
            "rect_vs_gt": rect_vs_gt, "coverage": coverage(run, faces), "tops": wall_tops(W, gti["faces"], matches),
            "gt": gti, "faces": faces}


# ---------------------------------------------------------------- plan figure
def plan_figure(D: dict, path: Path) -> None:
    """Top view in the room's own axes: ray density, kept planes, closed outline, GT faces dashed."""
    C, W, H = D["C"], D["W"], D["H"]
    th = np.radians(C["axis_deg"])
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])      # floor -> room axes
    P = np.load(RAYS)["P"]
    P = P[(P[:, 2] > 0.3) & (P[:, 2] < 2.2), :2].astype(np.float64) @ R.T
    out = np.array([q for w in H["walls"] for q in w["footprint_floor"]]) @ R.T
    lo, hi = out.min(0) - 0.5, out.max(0) + 0.5
    k = 110
    Wd, Ht = int((hi[0] - lo[0]) * k), int((hi[1] - lo[1]) * k)
    S = lambda p: ((p[0] - lo[0]) * k, Ht - (p[1] - lo[1]) * k)
    grid = np.zeros((Ht, Wd)); ij = np.floor((P - lo) * k).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < Wd) & (ij[:, 1] >= 0) & (ij[:, 1] < Ht)
    np.add.at(grid, (Ht - 1 - ij[ok, 1], ij[ok, 0]), 1)
    shade = (1 - np.clip(np.log1p(grid) / np.log1p(np.percentile(grid[grid > 0], 99)), 0, 1) * 0.55) * 255
    img = Image.fromarray(np.dstack([shade] * 3).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype(FONT, 15)
    for w in H["walls"]:
        poly = [S(q) for q in (np.asarray(w["footprint_floor"]) @ R.T)]
        dr.polygon(poly, fill=(22, 25, 29, 60) if w["role"] == "exterior" else (98, 80, 214, 70))
    for f in D["faces"]:                                                    # GT faces, dashed red
        c, u = np.asarray(f["c"]), np.asarray(f["u"])
        a0, a1 = f["a"]
        n = max(2, int((a1 - a0) / 0.08))
        for i in range(0, n, 2):
            p0 = (c + (a0 + (a1 - a0) * i / n) * u) @ R.T; p1 = (c + (a0 + (a1 - a0) * min(i + 1, n) / n) * u) @ R.T
            dr.line([S(p0), S(p1)], fill=(227, 73, 72), width=3)
    for r in W["walls"]:                                                    # kept planes, solid blue
        c, u = np.asarray(r["point"]), np.asarray(r["along"])
        p0, p1 = (c + r["a"][0] * u) @ R.T, (c + r["a"][1] * u) @ R.T
        dr.line([S(p0), S(p1)], fill=(42, 120, 214), width=5)
        mid = S((p0 + p1) / 2)
        dr.text((mid[0] + 6, mid[1] - 20), r["id"], font=font, fill=(42, 120, 214))
    for w in H["walls"]:
        a, b = np.asarray(w["start_floor"]) @ R.T, np.asarray(w["end_floor"]) @ R.T
        mid = S((a + b) / 2 - 0.35 * (np.asarray(w["normal_to_back"]) @ R.T))
        dr.text(mid, w["id"], font=font, fill=(22, 25, 29), anchor="mm")
    dr.line([(20, Ht - 18), (20 + k, Ht - 18)], fill=(69, 76, 85), width=3)
    dr.text((26 + k, Ht - 28), "1 m", font=font, fill=(69, 76, 85))
    img.save(path, optimize=True)


# ---------------------------------------------------------------- page pieces
def bars(rows, vmax=1.0, fmt=pct, ref=None, ref_label="", unit=""):
    """Horizontal single-series bars: [(label, value, note)]. Values are labelled; hover shows the note."""
    out = ['<div class="bars">']
    for label, v, note in rows:
        w = 0 if v is None else max(0.0, min(1.0, v / vmax)) * 100
        val = "—" if v is None else fmt(v)
        out.append(f'<div class="bar-row" title="{esc(note)}"><span class="bl">{esc(label)}</span>'
                   f'<span class="bt">{"" if ref is None else f"<i class=ref style=left:{ref / vmax * 100:.1f}%></i>"}'
                   f'<b style="width:{w:.1f}%"></b></span><span class="bv">{val}{unit}</span></div>')
    if ref is not None:
        out.append(f'<div class="bar-ref-note">竖线：{esc(ref_label)}</div>')
    out.append("</div>")
    return "".join(out)


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def kind(k):
    return {"measured": '<span class="k m">实测</span>', "assumed": '<span class="k a">假定</span>',
            "tuned": '<span class="k t">调参</span>', "gt": '<span class="k g">用到 GT</span>'}[k]


def step(no, title, sub, body, io_=None):
    io_html = ""
    if io_:
        io_html = '<div class="io">' + "".join(f'<div><span>{esc(k)}</span>{v}</div>' for k, v in io_) + "</div>"
    return f'<section id="s{no}"><h2><span class="no">{no:02d}</span>{title}</h2><p class="sub">{sub}</p>{io_html}{body}</section>'


def page(D: dict) -> str:
    seg, P, W, C, H, Fr = D["seg"], D["purity"], D["W"], D["C"], D["H"], D["F"]
    run = D["run"]
    kept = seg["kept"]
    nk = sum(kept.values())
    walls_kept = W["walls"]
    op = [(w["id"], o) for w in H["walls"] for o in w["openings"]]
    gt_open = [i for i in D["gt"]["items"] if i["label"] in ("door", "window", "opening")]
    rect = C["rectangle"]
    # ---------------------------------------------------------------- 00 overview
    flow = [("01", "2D 结构件分割", "GPT 点选 + SAM3", f"{nk} 个 mask"),
            ("02", "交付坐标系", "地面 mask 拟合", "amb3r_floor"),
            ("03", "抬升", "mask 像素 → AMB3R 点", f"{D['points'] / 1e6:.2f} M 点"),
            ("04", "射线", "全部帧每 4 个像素", f"{D['rays'] / 1e6:.2f} M 条"),
            ("05", "墙面", "局部平面 → 墙线 → 射线切开", f"{len(walls_kept)} 面墙 · {len(W['walls'][0:0]) + sum(len(r['openings']) for r in walls_kept)} 个开口"),
            ("06", "封闭", "内外墙 · 长方形 · 延伸", f"{rect['width_m']:.2f} × {rect['depth_m']:.2f} m"),
            ("07", "交接", "walls/v2", f"{len(H['walls'])} 面墙 · {sum(len(w['key_frames']) for w in H['walls'])} 张关键帧")]
    flow_html = '<div class="flow">' + "".join(
        f'<a class="fs" href="#s{int(n)}"><span class="fn">{n}</span><b>{t}</b><span class="fd">{d}</span><span class="fo">{o}</span></a>'
        for n, t, d, o in flow) + "</div>"
    files = table(["产物", "位置", "谁读"], [
        ["2D 分割结果", f"<code>artifacts/structure/{run}/segmentation/</code>", "floor_frame、lift、交接（关键帧 mask）"],
        ["交付坐标系", "<code>artifacts/scenes/scene0000_00/amb3r_floor.json</code>", "所有后续步骤、物体抬升"],
        ["点 / 射线", f"<code>points.npz</code> · <code>artifacts/structure/scene_rays.npz</code>", "walls、close"],
        ["墙面平面", "<code>walls.json</code> · <code>geom.npz</code>", "close、交接"],
        ["封闭结果", "<code>closed.json</code>", "交接"],
        ["墙体交接", "<code>artifacts/handoff/scene0000_00/walls_harness/</code>", "墙体生成模块"]])
    s0 = (f'<section id="s0"><h2><span class="no">00</span>一页看全</h2>'
          f'<p class="sub">RGB 视频 + AMB3R 的逐帧点图和位姿 → 给墙体生成模块的墙体交接文件。全程不读 ScanNet 标注；GT 只在本页的对照列里出现。</p>'
          f'{flow_html}<h3>每一步写什么文件</h3>{files}'
          f'<p class="note">标签说明：{kind("measured")} 由数据测出 · {kind("assumed")} 人为给定，交接文件里标 assumed · '
          f'{kind("tuned")} 手调的阈值 · {kind("gt")} 用到了 GT。</p></section>')

    # ---------------------------------------------------------------- 01 segmentation
    cfg = seg["cfg"]
    per = P["per"]
    pur_bars = bars([(f"{ZH[c]}（{per[c]['n']}）", per[c]["mean"], f"按面积加权 {pct(per[c]['area_weighted'])}") for c in LABELS if c in per],
                    ref=0.9, ref_label="90%")
    det = table(["类别", "保留的 mask", "出现该类的帧（GT）", "有该类 mask 的帧", "平均纯度", "按面积加权"],
                [[ZH[c], per[c]["n"], per[c]["frames_gt"], per[c]["frames_kept"], pct(per[c]["mean"]), pct(per[c]["area_weighted"])]
                 for c in LABELS if c in per])
    badrows = [[x["fid"], ZH[x["label"]], esc(x["pid"]), esc(x["where"][:48]), pct(x["purity"]),
                f'{pct(x["worst"][1])} 落在 {esc(str(x["worst"][0]))}'] for x in P["bad"]]
    ex = "".join(
        f'<figure class="ex"><div class="tri"><img src="{SEG_PAGE_ASSETS.format(run=run)}/{fid}_rgb.jpg" alt="原图 {fid}">'
        f'<img src="{SEG_PAGE_ASSETS.format(run=run)}/{fid}_kept.jpg" alt="保留的 mask {fid}">'
        f'<img src="{SEG_PAGE_ASSETS.format(run=run)}/{fid}_gt.jpg" alt="GT {fid}"></div>'
        f'<figcaption>帧 {fid}：原图 · 保留的 mask（按类别上色） · GT（只用于检查）</figcaption></figure>'
        for fid in EXAMPLE_FRAMES if (DOCS / SEG_PAGE_ASSETS.format(run=run) / f"{fid}_rgb.jpg").is_file())
    calls = ", ".join(f"{k} 次 × {v} 帧" for k, v in sorted(seg["segment_calls"].items()))
    s1_body = f"""
<p>一次对话看连续 {cfg['window']} 帧（每 2 帧取 1 帧，共 {seg['frames']} 帧），模型先在几帧之间判断一大片平面是什么，
再逐帧点正负点让 SAM3 出 mask，看结果决定保留、重做或放弃。只有四个标签：{"、".join(f"<code>{c}</code>" for c in LABELS)}。
规则写在 <code>src/structure/prompts/structure_v4.txt</code>，核心是“不怕漏，怕污染”：把柜子侧面标成墙会变成一面一致的假墙，后面没有办法发现；漏标则会被别的帧补上。</p>
<div class="kpis"><div><b>{nk}</b><span>保留的 mask</span></div><div><b>{seg['windows']}</b><span>对话（每个 {min(seg['rounds'])}–{max(seg['rounds'])} 轮）</span></div>
<div><b>{seg['prompt_tokens'] / 1e6:.2f} M</b><span>prompt token</span></div><div><b>{seg['seconds'] / 60:.0f} 分钟</b><span>墙钟时间</span></div>
<div><b>{len(seg['not_planned'])}</b><span>条“有意不标”的说明</span></div></div>
<h3>和 GT 比：每个 mask 有多少像素落在对的类别上</h3>
<p class="note">GT 是 ScanNet 的 2D 实例标注，只用于检查。门框像素落在墙或门的 mask 里算对。</p>
{pur_bars}{det}
<h3>纯度低于 75% 的 mask（{len(P['bad'])} 个）</h3>
{table(["帧", "类别", "计划", "模型写的位置", "纯度", "最多的错类"], badrows)}
<p>地面漏标的帧：{"、".join(f"{fid}（画面 {pct(s)} 是地面）" for fid, s in P["missed_floor"]) or "无"}。每帧 SAM 调用次数：{calls}。</p>
<div class="exs">{ex}</div>
<p class="more">逐帧看全部 50 帧：<a href="structure-{run}.html">structure-{run}.html</a></p>"""
    s1 = step(1, "2D 结构件分割", f"src/structure/segment/window.py · {esc(cfg['model'])} · prompt {cfg['prompt_sha256'][:8]}", s1_body,
              [("输入", f"{seg['frames']} 帧 RGB（100 帧采样里每隔一帧）"), ("输出", "每帧 result.json + mask"),
               ("关键设定", f"{kind('tuned')} 窗口 {cfg['window']} 帧、每对话最多 {cfg['max_rounds']} 轮、隔帧取样")])

    # ---------------------------------------------------------------- 02 frame
    ff = Fr["floor_fit"]; cams = Fr["cameras_above_floor_m"]
    s2_body = f"""
<p>用上一步全部 <code>floor</code> mask 的像素（{ff['points'] / 1e6:.2f} M 点，{ff['masks']} 个 mask）拟合一个平面，
法向朝相机那一侧就是 +Z，地面就是 z = 0；原点在第一帧相机脚下，+X 是第一帧相机的水平朝向。不读 ScanNet 的 axisAlignment 和 GT 位姿。
定义见 <code>docs/amb3r-floor-frame.md</code>，物体链也用同一个坐标系。</p>
{table(["量", "值", ""], [
        ["地面拟合残差", f"中位数 {f2(ff['residual_median_cm'])} cm，p90 {f2(ff['residual_p90_cm'])} cm，{pct(ff['within_3cm'])} 在 3 cm 内", kind("measured")],
        ["相机离地", f"中位数 {f2(cams['median'])} m，最低 {f2(cams['min'])} m，{cams['below_floor']} 帧在地面下", kind("measured")],
        ["第一帧", f"离地 {f2(Fr['first_frame']['height_m'])} m，俯视 {f1(-Fr['first_frame']['pitch_deg'])}°", kind("measured")],
        ["尺度", f"{Fr['scale']['value']:.4f}（{Fr['scale']['source']}）：AMB3R 相机中心对 GT 相机中心拟合出的尺度，只用尺度", kind("gt")]])}"""
    s2 = step(2, "交付坐标系 amb3r_floor", "src/geometry/floor_frame.py", s2_body,
              [("输入", "地面 mask + AMB3R 点图和位姿"), ("输出", "amb3r_floor.json（每个场景一份）")])

    # ---------------------------------------------------------------- 03 lift, 04 rays
    lift_rows = [[ZH[c], f"{D['lift'][c]:,}"] for c in LABELS]
    s3_body = f"""<p>每个 mask 像素取 AMB3R 点图在该像素的 3D 点（置信度 ≥ 0.3），转到交付坐标系；同时从点图网格求法向（朝相机）。一个像素最多一个点。</p>
{table(["类别", "点数"], lift_rows)}"""
    s3 = step(3, "抬升", "src/structure/lift.py", s3_body,
              [("输入", "2D mask"), ("输出", f"points.npz（{D['points']:,} 点）")])
    s4_body = f"""<p>不看 mask，把全部 {D['ray_frames']} 帧每隔 4 个像素取一条“相机 → 表面”的射线，共 {D['rays']:,} 条。
射线经过的地方是空的、终点处有东西。后面所有“这里有没有墙”的判断都靠它，而不是靠调出来的距离阈值。</p>"""
    s4 = step(4, "射线", "src/structure/rays.py", s4_body,
              [("输入", "AMB3R 点图和位姿（全部帧）"), ("输出", "scene_rays.npz（与 run 无关）")])

    # ---------------------------------------------------------------- 05 walls
    pr = W["params"]
    wrows = []
    for r in walls_kept:
        m = D["matches"].get(r["id"])
        wrows.append([r["id"], f"{r['length_m']:.2f} m", f"{r['area_m2']:.2f} m²", len(r["frames"]), f"{r['rms_cm']:.1f} cm",
                      f"{m['face']} {esc(m['name'])}" if m else "—", f"{m['offset_cm']:+.1f} cm" if m else "—"])
    op_rows = [[wid, f"{o['width_m']:.2f} × {o['height_m']:.2f} m", f"{o['z_floor'][0]:.2f}–{o['z_floor'][1]:.2f} m",
                " / ".join(f"{k} {v}" for k, v in o["bounds"].items())] for wid, o in op]
    gt_rows = [[esc(i["name"]), f"{i['width_m']:.2f} × {i['height_m']:.2f} m", f"{i['z'][0]:.2f}–{i['z'][1]:.2f} m", i["host"]] for i in gt_open]
    s5_body = f"""
<p>假设墙是竖直平面。四小步：</p>
<ol class="steps">
<li><b>局部平面</b>：一个 mask 一个 mask 地在俯视图里找直线（RANSAC），一个跨墙角的 mask 会按法向拆成两组。共 {W['local_planes']} 个局部平面。</li>
<li><b>墙线</b>：朝向差 ≤ {pr['facing_deg']:.0f}°、前后距离能串起来（≤ {pr['drift_m']} m）的局部平面算一条线，不管沿墙隔多远。共 {W['lines']} 条。</li>
<li><b>射线切开</b>：沿每条线做 10 cm 网格，一整列从下到上都被射线穿过、又没有实体的地方把线切断。共切了 {len(W['cuts'])} 刀。</li>
<li><b>留下墙面</b>：每段支持面积 ≥ {pr['min_area_m2']} m²、至少 {pr['min_frames']} 帧看到的留下：{len(walls_kept)} 面保留，{len(W['rejected'])} 面丢弃。
墙面上被射线穿过、左右上都有实墙围住的区域是开口。</li></ol>
<h3>保留的墙面和 GT</h3>{table(["墙面", "长度", "支持面积", "帧", "平面残差", "对应 GT 墙面", "比 GT 靠里"], wrows)}
<p class="note">“比 GT 靠里”是墙面相对 GT 墙面沿法向的偏移，正数表示往房间里偏。AMB3R 重建出的墙面本身就整体往里偏几厘米，所以这列主要是重建误差，不是拟合误差。</p>
<h3>开口：测到 {len(op)} 个，GT 里 {len(gt_open)} 个</h3>
{table(["墙", "大小", "离地", "四边"], op_rows)}
{table(["GT", "大小", "离地", "所在 GT 墙面"], gt_rows)}
<p>射线只能测出开着的洞。关着的门扇、玻璃窗会挡住射线，所以入户门和两扇窗测不到，尽管 2D 分割已经标出了门扇和窗。</p>
<h3>参数</h3>
{table(["参数", "值", "作用", ""], [
        ["DRIFT", f"{pr['drift_m']} m", "同一面墙在不同帧里的深度差上限；结果在 0.15–0.70 m 之间不变", kind("tuned")],
        ["FACING_DEG", f"{pr['facing_deg']}°", "朝向差多少以内算同一条墙线", kind("tuned")],
        ["CELL / MIN_RAYS", f"{pr['cell_m']} m / {pr['min_rays']}", "射线网格和一格要几条射线才算数；开口边界精度就是这一格", kind("tuned")],
        ["MIN_AREA / MIN_FRAMES", f"{pr['min_area_m2']} m² / {pr['min_frames']}", "一面墙至少要多少支持", kind("tuned")],
        ["MIN_OPENING", f"{pr['min_opening_m2']} m²", "多大的穿透区域算开口", kind("tuned")]])}"""
    s5 = step(5, "墙面", "src/structure/walls.py", s5_body,
              [("输入", "墙的点 + 射线"), ("输出", "walls.json（平面、开口、切口）")])

    # ---------------------------------------------------------------- 06 close
    bs = bars([(f"{r['id']}（{'内' if r['role'] == 'interior' else '外'}）", r["behind_seen"], "墙背后 0.15–0.8 m 的地面上被射线看到的比例")
               for r in C["walls"]], ref=0.5, ref_label="超过 50% 判为内墙")
    rv = table(["长方形的边", "来自", "对应 GT 墙面", "比 GT 靠里"],
               [[x["side"], "观测墙" if x["from"] == "observed" else "看到的空间", f"{x['face']} {esc(x['name'])}", f"{x['inward_cm']:+.1f} cm"]
                for x in D["rect_vs_gt"]])
    hid = {w["evidence"]["planes"][0]: w["id"] for w in H["walls"] if w["role"] == "interior"}
    fin = table(["墙", "类型", "长度", "观测", "推断"],
                [[f["id"] if f["role"] == "exterior" else f"{hid.get(f['id'], f['id'])}（{f['id']}）", "外墙" if f["role"] == "exterior" else "内墙", f"{f['length_m']:.2f} m",
                  f"{f['length_m'] - f['inferred_m']:.2f} m", f"{f['inferred_m']:.2f} m"] for f in C["final"]])
    tops = table(["墙面", "射线看到的墙面最高处", "GT 墙顶", "统一层高"],
                 [[t["id"], f"{t['seen_top']:.2f} m" if t["seen_top"] else "—", f"{t['gt_top']:.2f} m" if t["gt_top"] else "—",
                   f"{C['storey']['height_m']:.2f} m"] for t in D["tops"]])
    s6_body = f"""
<ol class="steps">
<li><b>内墙还是外墙</b>：外墙背后是房子外面，任何一帧都看不到；内墙背后是别的房间，从门洞能看到。量每面墙背后那条地带被射线看到的比例：</li></ol>
{bs}
<ol class="steps" start="2">
<li><b>主方向</b>：墙的朝向按面积加权、按 90° 取平均，得 {C['axis_deg']:+.2f}°；偏差 ≤ 10° 的墙转到主方向上。</li>
<li><b>长方形</b>：每条边放在朝这边的外墙上；没有外墙的边放在“看到的空间”边缘。看到的空间有 {pct(rect['seen_outside_share'])} 落在长方形外，长方形内 {pct(rect['inside_seen_share'])} 被看到。</li>
<li><b>内墙延伸</b>：内墙两端沿自身延长，到 {C['params']['max_ext_m']} m 内遇到的第一条墙线为止；如果延长段被射线穿过就不延。</li>
<li><b>层高和厚度</b>：层高 {C['storey']['height_m']:.2f} m {kind('assumed')}，外墙 {C['thickness']['exterior_m'] * 100:.0f} cm / 内墙 {C['thickness']['interior_m'] * 100:.0f} cm {kind('assumed')}，厚度加在墙背后，房间一侧的墙面不动。</li></ol>
<figure class="plan"><img src="assets/structure-overview/plan.png" alt="俯视图">
<figcaption>俯视图，按房间主方向摆正。灰色：射线终点密度（0.3–2.2 m 高）；蓝色实线：保留的墙面；深灰 / 紫色块：交接里的外墙 / 内墙占地；红色虚线：GT 墙面。</figcaption></figure>
<h3>长方形和 GT</h3>{rv}
{table(["方向", "长方形", "GT（两面对应墙之间）", "差", "对应的 GT 墙面"],
       [[x["axis"], f"{x['ours']:.2f} m", f"{x['gt']:.2f} m", f"{(x['ours'] - x['gt']) * 100:+.0f} cm", esc(x["faces"])] for x in D["size_vs_gt"]])}
<p class="note">交接里的 W/S/E/N 按 amb3r_floor 里房间的主方向起名，和 ScanNet GT 页面上的“南墙、北墙”不是同一套方位：交接的 ext_W 落在 GT 的北墙上。</p>
<h3>封闭后每面墙观测了多少</h3>{fin}
<h3>层高</h3>{tops}
<p class="note">相机最高只往上看约 24°，所以“看到的最高处”在有些墙上只是视野上限，不一定是墙顶。</p>"""
    s6 = step(6, "封闭", "src/structure/close.py", s6_body,
              [("输入", "墙面平面 + 射线"), ("输出", "closed.json（外墙轮廓、内墙、层高、厚度）")])

    # ---------------------------------------------------------------- 07 handoff
    hrows = [[w["id"], "外墙" if w["role"] == "exterior" else "内墙", f"{w['length_m']:.2f} m",
              f"{w['evidence']['observed_m']:.2f} / {w['evidence']['inferred_m']:.2f} m",
              "、".join(f"{k['source_frame_id']}（{k['view_angle_deg']:.0f}°, {k['reprojection_px']:.0f} px）" for k in w["key_frames"]) or "—",
              len(w["openings"])] for w in H["walls"]]
    s7_body = f"""<p>每面墙给房间一侧墙面的线（房间在左手边）、背面方向、厚度、层高、占地四边形（转角已处理）、观测/推断分段、开口和最多 3 张关键帧（原图、按墙切开的 mask、预览）。坐标系 amb3r_floor，和物体交接相同。</p>
{table(["墙", "类型", "长度", "观测 / 推断", "关键帧（视角, 重投影）", "开口"], hrows)}
<p class="more">交接文件预览页：<code>site/handoff/scene0000_00/walls_harness/index.html</code>（<code>python -m pagegen.build_walls_handoff</code>）；三维工作台：<a href="structure-geom-3d.html">structure-geom-3d.html</a></p>"""
    s7 = step(7, "交接", "src/handoff/walls_from_harness.py · walls/v2", s7_body,
              [("输入", "closed.json + walls.json + 关键帧 mask"), ("输出", "walls.json + keyframes/ + README")])

    # ---------------------------------------------------------------- 08 coverage
    cov = D["coverage"]
    cov_rows = [[f"{c['key']} {esc(c['name'])}", f"{c['length_m']:.2f} m", pct(c["rays"]), pct(c["masks"]), pct(c["kept"])] for c in cov]
    s8_body = f"""<p>每面 GT 墙面按 10 cm × 10 cm 分格（0.1–2.3 m 高），看每一层“看到”了多少格。从射线到 mask 掉得最多，说明瓶颈在 2D 分割的覆盖面，不在几何。</p>
{table(["GT 墙面", "长度", "有射线终点", "有墙 mask 点", "进了某个平面"], cov_rows)}
<p class="note">“有射线终点”包括墙前家具挡住的部分，所以不会到 100%。西墙（F6）大半被窗帘和衣柜挡着，按规则不该标成墙。</p>"""
    s8 = step(8, "一层层掉了多少", "GT 只用于这张表", s8_body)

    # ---------------------------------------------------------------- 09 weak points
    tops_ok = [t for t in D["tops"] if t["seen_top"] and t["gt_top"]]
    weak = [
        ("开口只有射线能测到的", f"测到 {len(op)} 个，GT {len(gt_open)} 个。关着的门、玻璃窗挡住射线；2D 分割已经标出的门扇（5 帧）和窗（3 帧）没有用来生成开口。",
         "门扇 / 窗 mask 落在墙面上的范围生成开口（source: mask），同样要求四边有界。要先定：门窗的洞是否交给下游。"),
        ("层高统一假定 2.8 m", "GT 墙顶 2.33–2.91 m（天花板是斜的）。" + "；".join(
            f"{t['id']} 看到 {t['seen_top']:.2f} / GT {t['gt_top']:.2f}" for t in tops_ok[:4]) + "。",
         "每面墙判断墙顶是否被天花板封住（和开口四边同一套射线判断），封住的用实测高度，没有的才用 2.8 m。"),
        ("房间整体偏小", "；".join(f"{x['axis']} 方向小 {(x['gt'] - x['ours']) * 100:.0f} cm" for x in D["size_vs_gt"]) + "。各边比 GT 靠里："
         + "、".join(f"{x['side']} {x['inward_cm']:+.0f} cm" for x in D["rect_vs_gt"])
         + "。一部分是重建本身的内缩（墙面平均往里几厘米），最大的那一边只看到一小段墙，而它外面还有射线终点。",
         "每条边取“墙面拟合位置”和“最外侧一条密集的射线终点带”中靠外的那个，满足“长方形尽量包住所有点云”。"),
        ("卫生间整体错位，被并进同一条墙线", "卫生间里拍的几帧被 AMB3R 整体往北偏了 10–20 cm；墙线合并允许 30 cm 的深度漂移，于是卫生间南墙并进了主屋南墙，平面被拉偏几厘米，两条厚度线索也只剩一半。",
         "合并时如果两段的前后距离明显超过单面墙的深度波动，就不合并，或者在 evidence 里标出来。"),
        ("2D 分割只覆盖一部分墙面", "墙 mask 盖住的 GT 墙面格子明显少于射线看到的（见上一节）；50 帧只用了 100 帧的一半；窗的纯度只有 " + (pct(per["window"]["mean"]) if "window" in per else "—") + "。",
         "先量清楚：是帧数不够，还是模型在这些帧里主动没标（24 条“有意不标”的说明）。"),
        ("没有固定的 GT 评估", "这页的 GT 列是每次临时算的；参数改了没法一眼看出哪里变好、哪里变坏。",
         "把这页的对照固化成 tools/structure/gt_eval.py，每次改动都跑，结果写成一行可比较的指标。"),
        ("尺度来自 GT", f"{Fr['scale']['source']}：换场景时没有 GT 位姿。metric head 原始尺度偏大约 12.7%。",
         "用门高、门宽这类已知尺寸估尺度；和物体链一起做。"),
        ("开口边界只有 ±10 cm", "射线网格 10 cm 一格；换坐标系时开口从 0.60 × 1.90 变成 0.70 × 1.80 m（GT 0.63 × 1.95）。",
         "开口边界在格子定出大致范围后，再用穿过 / 停在墙面的射线逐条细化到厘米级。"),
    ]
    wk = "".join(f'<div class="weak"><div class="wn">{i + 1:02d}</div><div><h4>{esc(t)}</h4><p>{esc(a)}</p><p class="fix"><b>可以怎么改：</b>{esc(b)}</p></div></div>'
                 for i, (t, a, b) in enumerate(weak))
    s9 = (f'<section id="s9"><h2><span class="no">09</span>已知的薄弱点</h2>'
          f'<p class="sub">按对交接质量的影响排序；每条给出现状的实测和一个改法，作为下一步梳理的起点</p>{wk}</section>')

    rail = "".join(f'<a href="#s{i}"><span class="n">{i:02d}</span>{t}</a>' for i, t in enumerate(
        ["一页看全", "2D 分割", "交付坐标系", "抬升", "射线", "墙面", "封闭", "交接", "一层层掉了多少", "薄弱点"]))
    return PAGE.format(rail=rail, run=esc(run), body=s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9,
                       nk=nk, nw=len(H["walls"]), rect=f"{rect['width_m']:.2f} × {rect['depth_m']:.2f} m",
                       tok=f"{seg['prompt_tokens'] / 1e6:.2f}")


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>结构件识别方案 · scene0000_00 · {run}</title>
<style>
:root{{--ink:#16191d;--ink-2:#454c55;--ink-3:#6b7480;--line:#dfe3e8;--line-2:#eef1f4;--bg:#fbfcfd;--card:#fff;
 --accent:#0f766e;--accent-soft:#e6f4f1;--bar:#2a78d6;--warn:#b45309;--warn-soft:#fef6e7;--model:#7c3aed;--model-soft:#f3eefe;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15.5px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif}}
.wrap{{display:grid;grid-template-columns:minmax(0,1fr);max-width:1240px;margin:0 auto;padding:0 28px 120px}}
.rail{{display:none}} main{{min-width:0}}
@media(min-width:1080px){{.wrap{{grid-template-columns:200px minmax(0,1fr);gap:48px}} .rail{{position:sticky;top:0;align-self:start;padding:44px 0;display:block}}}} .rail a{{display:block;padding:6px 10px;border-radius:6px;color:var(--ink-2);text-decoration:none;font-size:13.5px}}
.rail a:hover{{background:var(--line-2)}} .rail .n{{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-right:7px}}
header{{padding:52px 0 26px;border-bottom:1px solid var(--line)}}
.eyebrow{{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}}
h1{{margin:8px 0 12px;font-size:34px;line-height:1.25}} .lede{{max-width:70ch;color:var(--ink-2);margin:0}}
.tags{{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px}} .tags span{{font-family:var(--mono);font-size:12px;border:1px solid var(--line);background:var(--card);border-radius:5px;padding:2px 9px}}
section{{padding-top:52px;scroll-margin-top:16px}} h2{{margin:0 0 4px;font-size:22px;display:flex;gap:12px;align-items:baseline}}
h2 .no{{font-family:var(--mono);font-size:13px;color:var(--accent)}} .sub{{margin:0 0 18px;color:var(--ink-3);font-family:var(--mono);font-size:13px}}
h3{{margin:28px 0 10px;font-size:16px}} h4{{margin:0 0 4px;font-size:15.5px}} p{{margin:0 0 12px}} a{{color:var(--accent)}}
code{{font-family:var(--mono);font-size:.86em;background:var(--line-2);padding:1px 5px;border-radius:4px}}
.note{{color:var(--ink-3);font-size:13.5px}} .more{{font-size:14px}}
.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 6px}}
.fs{{display:flex;flex-direction:column;gap:2px;padding:12px 14px;background:var(--card);border:1px solid var(--line);border-radius:10px;text-decoration:none;color:var(--ink);position:relative}}
.fs:hover{{border-color:var(--accent)}} .fn{{font-family:var(--mono);font-size:11px;color:var(--accent)}} .fs b{{font-size:15px}}
.fd{{font-size:12.5px;color:var(--ink-3)}} .fo{{font-family:var(--mono);font-size:12.5px;color:var(--ink-2);margin-top:4px}}
.io{{display:flex;flex-wrap:wrap;gap:8px 22px;margin:0 0 16px;padding:10px 14px;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:13.5px}}
.io span{{color:var(--ink-3);margin-right:8px;font-size:12px}}
.tw{{overflow-x:auto;margin:6px 0 14px}} table{{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card)}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line-2);vertical-align:top}} th{{color:var(--ink-3);font-weight:600;font-size:12.5px;border-bottom-color:var(--line)}}
.k{{font-size:11.5px;padding:1px 7px;border-radius:10px;white-space:nowrap;border:1px solid}}
.k.m{{color:var(--accent);background:var(--accent-soft);border-color:#b7ddd6}} .k.a{{color:var(--warn);background:var(--warn-soft);border-color:#f2d6a7}}
.k.t{{color:var(--ink-2);background:var(--line-2);border-color:var(--line)}} .k.g{{color:var(--model);background:var(--model-soft);border-color:#ddd0fb}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 10px}} .kpis div{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;min-width:120px}}
.kpis b{{display:block;font-size:20px;font-family:var(--mono)}} .kpis span{{font-size:12.5px;color:var(--ink-3)}}
.bars{{margin:8px 0 14px;max-width:640px}} .bar-row{{display:grid;grid-template-columns:150px 1fr 70px;gap:10px;align-items:center;padding:3px 0}}
.bl{{font-size:13px;color:var(--ink-2)}} .bv{{font-family:var(--mono);font-size:12.5px;color:var(--ink)}}
.bt{{position:relative;height:14px;background:var(--line-2);border-radius:2px}} .bt b{{display:block;height:100%;background:var(--bar);border-radius:0 4px 4px 0}}
.bt .ref{{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink-3)}} .bar-row:hover .bt b{{filter:brightness(1.12)}}
.bar-ref-note{{font-size:12px;color:var(--ink-3)}}
.exs{{display:grid;gap:14px;margin:10px 0}} .tri{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}} .tri img{{width:100%;border-radius:6px;border:1px solid var(--line)}}
figcaption{{font-size:12.5px;color:var(--ink-3);margin-top:4px}} .plan img{{max-width:100%;border:1px solid var(--line);border-radius:8px}}
ol.steps{{padding-left:22px}} ol.steps li{{margin-bottom:6px}}
.weak{{display:grid;grid-template-columns:44px 1fr;gap:12px;padding:14px 0;border-bottom:1px solid var(--line-2)}}
.wn{{font-family:var(--mono);color:var(--accent);font-size:18px}} .fix{{color:var(--ink-2);font-size:14px}}
@media (prefers-color-scheme:dark){{:root{{--ink:#f0efec;--ink-2:#c3c2b7;--ink-3:#9a998f;--line:#34332f;--line-2:#262624;--bg:#1a1a19;--card:#211f1e;--bar:#3987e5;--accent:#2fb39f;--accent-soft:#16302c}}}}
</style>
</head>
<body>
<div class="wrap">
<nav class="rail">{rail}</nav>
<main>
<header>
<div class="eyebrow">structure recognition · scene0000_00</div>
<h1>结构件识别方案：从 RGB 到墙体交接</h1>
<p class="lede">把视频里的墙、地面、门、窗识别出来，求出墙的几何，封闭成一个房间，交给墙体生成模块。本页按流水线顺序讲每一步做了什么、在这个场景上的实测结果、哪些值是测出来的、哪些是假定或手调的，最后列出已知的薄弱点。所有数字由 <code>python -m pagegen.build_structure_overview</code> 从产物里读出。</p>
<div class="tags"><span>run {run}</span><span>{nk} 个 2D mask</span><span>{nw} 面墙</span><span>{rect}</span><span>{tok} M token</span><span>坐标系 amb3r_floor</span></div>
</header>
{body}
</main>
</div>
</body>
</html>
"""


def build(run: str, out: Path = OUT) -> None:
    D = collect(run)
    ASSETS.mkdir(parents=True, exist_ok=True)
    plan_figure(D, ASSETS / "plan.png")
    out.write_text(page(D))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB) and {ASSETS / 'plan.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="struct_v4_win5")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    build(a.run, a.out)


if __name__ == "__main__":
    main()
