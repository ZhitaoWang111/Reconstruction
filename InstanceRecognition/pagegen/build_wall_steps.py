"""The wall-extraction steps, one picture each: docs/structure-wall-steps.html.

    python -m src.structure.walls --run struct_v4_win5 --trace     # writes walls_trace.{npz,json}
    python -m pagegen.build_wall_steps --run struct_v4_win5

Everything is the top view in the room's own axes (the closed rectangle's main direction),
drawn from what src.structure.walls recorded while it ran - nothing is recomputed here:

  1 thin and keep the upright points         every wall point; the ones kept (normal near horizontal)
  2 planes inside each 2D mask               each local plane in its own colour; the sweep's planes marked
  3 join planes into wall lines              planes coloured by the line they joined
  4 cut lines where the rays see through     one line at a time: its ray grid along the wall x height
  5 keep or reject                           kept walls, rejected ones, with their openings

Data goes into the page itself (window.STEPS); about a megabyte.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from src import config
from src.structure.frames import OUT_ROOT

OUT = config.ROOT / "docs/structure-wall-steps.html"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def build(run: str, out: Path = OUT) -> None:
    d = OUT_ROOT / run
    t = json.loads((d / "walls_trace.json").read_text())
    z = np.load(d / "walls_trace.npz")
    W = json.loads((d / "walls.json").read_text())
    C = json.loads((d / "closed.json").read_text())
    masks = json.loads((d / "masks.json").read_text())
    th = np.radians(C["axis_deg"])
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])          # floor xy -> room axes
    rot = lambda p: (np.asarray(p, dtype=np.float64)[..., :2] @ R.T)
    P = rot(z["P"])
    lo, hi = P.min(0) - 0.3, P.max(0) + 0.3
    q = np.round((P - lo) / (hi - lo) * 65535).astype("<u2")
    lines = []
    for l in t["lines"]:
        c, n = np.asarray(l["c"]), np.asarray(l["n"]); u = np.array([-n[1], n[0]])
        lines.append({**l, "p0": rot(c + l["lo"] * u).tolist(), "p1": rot(c + l["hi"] * u).tolist(),
                      "n_room": (R @ n).tolist(), "c_room": rot(c).tolist(),
                      "dropped_small": [[rot(c + a * u).tolist(), rot(c + b * u).tolist()] for a, b in l.get("dropped_small", [])]})
    planes = []
    for r in W["walls"] + W["rejected"]:
        c, u = np.asarray(r["point"]), np.asarray(r["along"])
        planes.append({"id": r["id"], "kept": r["id"].startswith("W"), "line": r["line"], "why": r.get("why", ""),
                       "p0": rot(c + r["a"][0] * u).tolist(), "p1": rot(c + r["a"][1] * u).tolist(),
                       "n_room": (R @ np.asarray(r["normal"])).tolist(), "area_m2": r["area_m2"], "frames": len(r["frames"]),
                       "length_m": r["length_m"],
                       "openings": [{"p0": rot(o["corners"][0]).tolist(), "p1": rot(o["corners"][1]).tolist(),
                                     "w": o["width_m"], "h": o["height_m"]} for o in r["openings"]]})
    locs = []
    for L in t["locals"]:
        c, n = np.asarray(L["c"]), np.asarray(L["n"])
        sel = z["local"] == L["k"]
        u = np.array([-n[1], n[0]]); a = (z["P"][sel, :2] - c) @ u
        locs.append({**L, "p0": rot(c + a.min() * u).tolist(), "p1": rot(c + a.max() * u).tolist(),
                     "line": int(z["line"][sel][0]) if sel.any() else -1,
                     "where": masks[L["mask"]]["where"], "frame": masks[L["mask"]]["frame"]})
    data = {"run": run, "lo": lo.tolist(), "hi": hi.tolist(), "n": int(len(P)), "pos": b64(q),
            "horiz": b64(z["horiz"].astype(np.uint8)), "local": b64(z["local"].astype("<i2")),
            "line": b64(z["line"].astype("<i2")), "plane": b64(z["plane"].astype("<i2")), "mask": b64(z["mask"].astype("<u2")),
            "voxel_m": t["voxel_m"], "params": t["params"], "locals": locs, "lines": lines, "planes": planes,
            "reps_deg": t["reps_deg"], "axis_deg": C["axis_deg"],
            "counts": {"wall_points": W["wall_points"], "thinned": W["wall_points_thinned"],
                       "horizontal": W["wall_points_horizontal"], "unassigned": W["wall_points_unassigned"],
                       "wall_masks": sum(1 for m in masks if m["label"] == "wall")}}
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): {len(P)} points, {len(locs)} local planes, {len(lines)} lines, "
          f"{len(planes)} planes")


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>墙面提取，一步一步 · struct_v4_win5</title>
<style>
:root{--ink:#16191d;--ink-2:#454c55;--ink-3:#6b7480;--line:#dfe3e8;--line-2:#eef1f4;--bg:#fbfcfd;--card:#fff;--accent:#0f766e;
 --ok:#2a78d6;--rej:#8b9097;--none:#d6dade;--cut:#e34948;--free:#1baf7a;--solid:#2a78d6}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:36px 28px 80px} h1{font-size:28px;margin:0 0 8px} .lede{color:var(--ink-2);max-width:78ch;margin:0 0 6px}
.steps{display:flex;gap:6px;margin:22px 0 14px;flex-wrap:wrap} .steps button{font:inherit;border:1px solid var(--line);background:var(--card);border-radius:8px;padding:8px 14px;cursor:pointer;color:var(--ink-2);text-align:left}
.steps button b{display:block;font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--ink-3)} .steps button.on{border-color:var(--accent);background:#e6f4f1;color:var(--accent)}
.steps button.on b{color:var(--accent)}
.stage{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:16px}
.view{position:relative;background:var(--card);border:1px solid var(--line);border-radius:10px;height:640px;overflow:hidden}
canvas{width:100%;height:100%;display:block;cursor:grab}
.side{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;height:640px;overflow:auto;font-size:14px}
.side h3{margin:0 0 6px;font-size:16px} .side p{margin:0 0 10px;color:var(--ink-2)} .side .kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:8px 0 12px;font-size:13px}
.side .kv span:nth-child(odd){color:var(--ink-3)} .side .kv span:nth-child(even){font-family:ui-monospace,Menlo,monospace}
.side .lg div{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink-2);margin:2px 0}
.chip{width:12px;height:12px;border-radius:3px;flex:none;display:inline-block}
.ln{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0} .ln button{font:inherit;font-size:12px;border:1px solid var(--line);background:#fff;border-radius:5px;padding:1px 7px;cursor:pointer}
.ln button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.grid{margin-top:10px} .grid canvas{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;cursor:default}
.tip{position:absolute;pointer-events:none;background:rgba(22,25,29,.92);color:#fff;font-size:12px;padding:5px 8px;border-radius:6px;display:none;max-width:320px}
.hint{position:absolute;right:10px;bottom:8px;font-size:12px;color:var(--ink-3);background:rgba(255,255,255,.85);padding:2px 6px;border-radius:4px}
code{font-family:ui-monospace,Menlo,monospace;font-size:.88em;background:var(--line-2);padding:1px 5px;border-radius:4px}
</style></head><body><div class="wrap">
<h1>墙面提取，一步一步</h1>
<p class="lede">src/structure/walls.py 在这一轮（<code>struct_v4_win5</code>）上每一步实际做了什么。全部是俯视图，按房间主方向摆正，画的是程序跑的时候记下来的中间结果，没有重算。点上面的步骤切换；滚轮缩放，拖动平移，悬停看说明。</p>
<p class="lede" style="font-size:13px;color:var(--ink-3)">由 <code>python -m src.structure.walls --run struct_v4_win5 --trace</code> 和 <code>python -m pagegen.build_wall_steps</code> 生成。</p>
<div class="steps" id="steps"></div>
<div class="stage"><div class="view" id="view"><canvas id="cv"></canvas><div class="tip" id="tip"></div><div class="hint">滚轮缩放 · 拖动平移 · 双击回到全图</div></div>
<div class="side" id="side"></div></div>
</div>
<script>
window.STEPS = __DATA__;
(function(){
const D = window.STEPS, $ = (id) => document.getElementById(id);
const bytes = (b) => { const s = atob(b), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; };
const n = D.n, q = new Uint16Array(bytes(D.pos).buffer), P = new Float32Array(n * 2);
for (let i = 0; i < n * 2; i++) { const k = i % 2; P[i] = D.lo[k] + q[i] / 65535 * (D.hi[k] - D.lo[k]); }
const H = bytes(D.horiz), LOC = new Int16Array(bytes(D.local).buffer), LIN = new Int16Array(bytes(D.line).buffer), PL = new Int16Array(bytes(D.plane).buffer);
const NK = D.planes.filter((p) => p.kept).length;
// 12 distinguishable hues, assigned in fixed order; a colour names an identity on this step only
const PAL = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948","#0e9fb0","#a2620f","#7c8e00","#b04ac0"];
const col = (i) => PAL[((i % PAL.length) + PAL.length) % PAL.length];
const f1 = (x) => (+x).toFixed(1), f2 = (x) => (+x).toFixed(2);
const pr = D.params, c = D.counts;

const STEPS = [
 { key:"pts", t:"① 精简、只留竖直面的点", h:`<h3>① 精简墙点，只留竖直面上的点</h3>
   <p>2D 分割里标成“墙”的 ${c.wall_masks} 个 mask，抬成三维后共 ${c.wall_points.toLocaleString()} 个点。先在 ${D.voxel_m*100} cm 的小方格里每格只留一个（${c.thinned.toLocaleString()} 个，每个代表约 ${D.voxel_m*D.voxel_m*1e4} cm² 墙面）。</p>
   <p>再看每个点的法向（这块表面朝哪）：墙是竖直的，法向应该接近水平。偏离水平超过 ${pr.horizontal_deg}° 的点（家具顶面、天花板、墙角斜面）不参与后面的步骤。</p>`,
   lg:[["#2a78d6","参与后面步骤的点（法向接近水平）"],["#e34948","去掉的点（法向偏上或偏下）"]] },
 { key:"local", t:"② 每个 mask 里找平面", h:`<h3>② 在每个 mask 里找平面</h3>
   <p>按“来自哪个 2D mask”分组。俯视看，竖直的墙就是一条线：在组里反复随机挑两个点连线，找贴上最多点（${pr.tol_m*100} cm 内）的那条，作为一块墙面；拿掉它的点继续找，直到剩下的点凑不够 ${pr.min_local_m2} m²。</p>
   <p>一个 mask 跨过墙角时，先按法向拆成两组（夹角超过 ${pr.split_deg}°）。所有 mask 做完后，把没被收走的点合起来再找一遍（“扫尾”）。</p>
   <p>结果：${D.locals.length} 块小墙面，其中 ${D.locals.filter((l)=>l.sweep).length} 块来自扫尾。每块一种颜色（颜色循环使用，只区分相邻的块）；浅灰是没进任何一块的点。悬停看它来自哪个 mask。</p>`,
   lg:[["#d6dade","没进任何小墙面的点"],["#16191d","线段：小墙面的位置和长度，短刻度指向它朝的方向"]] },
 { key:"lines", t:"③ 连成墙线", h:`<h3>③ 把小墙面连成墙线</h3>
   <p>同一面墙会被不同 mask、不同帧拍成好几块。先按朝向分组（相差 ${pr.facing_deg}° 以内算一组，共 ${D.reps_deg.length} 组），组里按前后位置排好，相邻两块前后差不超过 ${pr.drift_m*100} cm 就连起来——<b>只看前后，不看左右</b>，沿墙隔多远都会连上。</p>
   <p>结果：${D.lines.length} 条墙线，每条一种颜色，细线是这条墙线的走向和长度。注意那几条横穿房间的长线：几块朝向相同、恰好共线但不相连的墙点被连在了一起，下一步要把它们切开。</p>`,
   lg:[["#d6dade","不属于任何墙线的点"]] },
 { key:"cut", t:"④ 用射线切开", h:"", lg:[] },
 { key:"keep", t:"⑤ 保留、丢弃、开口", h:`<h3>⑤ 留下够格的，找墙上的洞</h3>
   <p>切开后的每一段，支持面积至少 ${pr.min_area_m2} m²、至少 ${pr.min_frames} 帧看到，才留下来：保留 ${NK} 面（彩色，标 W），丢弃 ${D.planes.length-NK} 面（灰色虚线，标 R）。悬停看丢弃原因。</p>
   <p>在保留的墙上再看一遍射线网格：中间一块被射线穿透、左右上三边有墙围着，就是墙上的开口（黑色框，画在墙线上的那一段）。</p>
   <p>最后，墙点里有 ${c.unassigned.toLocaleString()} 个（${(100*c.unassigned/c.horizontal).toFixed(0)}%）没进任何保留或丢弃的墙面，浅灰显示。</p>`,
   lg:[["#d6dade","没进任何墙面的点"],["#8b9097","丢弃的墙面"],["#16191d","墙上的开口"]] },
];
let step = 0, selLine = D.lines.findIndex((l) => l.code.includes("O")), hover = null;
if (selLine < 0) selLine = 0;
$("steps").innerHTML = STEPS.map((s, i) => `<button data-i="${i}"><b>STEP ${i+1}</b>${s.t.slice(2)}</button>`).join("");
$("steps").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { step = +b.dataset.i; render(); } });

// ---------- view
const cv = $("cv"), ctx = cv.getContext("2d");
const V = { k: 1, ox: 0, oy: 0 };
function fit() { const r = cv.getBoundingClientRect(); const w = D.hi[0] - D.lo[0], h = D.hi[1] - D.lo[1];
  V.k = Math.min(r.width / w, r.height / h) * 0.94; V.ox = (D.lo[0] + D.hi[0]) / 2; V.oy = (D.lo[1] + D.hi[1]) / 2; }
const S = (x, y) => { const r = cv.getBoundingClientRect(); return [r.width / 2 + (x - V.ox) * V.k, r.height / 2 - (y - V.oy) * V.k]; };
const W2 = (sx, sy) => { const r = cv.getBoundingClientRect(); return [V.ox + (sx - r.width / 2) / V.k, V.oy - (sy - r.height / 2) / V.k]; };
function seg(a, b, color, width, dash) { const p = S(...a), q2 = S(...b); ctx.beginPath(); ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round"; ctx.moveTo(...p); ctx.lineTo(...q2); ctx.stroke(); ctx.setLineDash([]); }
function tick(a, b, nr, color) { const m = [(a[0]+b[0])/2, (a[1]+b[1])/2]; seg(m, [m[0]+nr[0]*0.18, m[1]+nr[1]*0.18], color, 2); }
function label(p, text, color) { const s = S(...p); ctx.font = "600 12px -apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif"; const w = ctx.measureText(text).width + 8;
  ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.fillRect(s[0] - w/2, s[1] - 9, w, 18); ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, s[0], s[1]); }
function dots(colorOf) {
  const r = Math.max(1, Math.min(3, V.k * D.voxel_m * 0.6));
  for (let i = 0; i < n; i++) { const c2 = colorOf(i); if (!c2) continue; const s = S(P[2*i], P[2*i+1]); ctx.fillStyle = c2; ctx.fillRect(s[0] - r/2, s[1] - r/2, r, r); }
}
function draw() {
  const dpr = window.devicePixelRatio || 1, r = cv.getBoundingClientRect();
  cv.width = r.width * dpr; cv.height = r.height * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, r.width, r.height);
  const k = STEPS[step].key;
  if (k === "pts") dots((i) => H[i] ? "#2a78d6" : "#e34948");
  if (k === "local") {
    dots((i) => !H[i] ? null : LOC[i] < 0 ? "#d6dade" : col(LOC[i]));
    D.locals.forEach((l) => { seg(l.p0, l.p1, "#16191d", hover && hover.kind === "local" && hover.i === l.k ? 3 : 1.2, l.sweep ? [4, 3] : null); });
  }
  if (k === "lines") {
    dots((i) => !H[i] ? null : LIN[i] < 0 ? "#d6dade" : col(LIN[i]));
    D.lines.forEach((l, i) => { seg(l.p0, l.p1, col(i), hover && hover.kind === "line" && hover.i === i ? 4 : 2); tick(l.p0, l.p1, l.n_room, col(i)); });
  }
  if (k === "cut") {
    dots((i) => !H[i] ? null : LIN[i] === selLine ? col(selLine) : "#e6e8eb");
    D.lines.forEach((l, i) => seg(l.p0, l.p1, i === selLine ? "#16191d" : "#c8ccd2", i === selLine ? 2.5 : 1));
    const l = D.lines[selLine], u = [l.p1[0] - l.p0[0], l.p1[1] - l.p0[1]], L = Math.hypot(...u);
    // cut columns along the line in red
    for (let j = 0; j < l.code.length; j++) if (l.code[j] === "O") {
      const a0 = j * l.cell / L, a1 = (j + 1) * l.cell / L;
      seg([l.p0[0] + u[0]*a0, l.p0[1] + u[1]*a0], [l.p0[0] + u[0]*a1, l.p0[1] + u[1]*a1], "#e34948", 6);
    }
    l.dropped_small.forEach(([a, b]) => seg(a, b, "#8b9097", 6, [2, 3]));
    D.planes.filter((p) => p.line === selLine).forEach((p) => { seg(p.p0, p.p1, p.kept ? "#2a78d6" : "#8b9097", 6); label([(p.p0[0]+p.p1[0])/2 - l.n_room[0]*0.35, (p.p0[1]+p.p1[1])/2 - l.n_room[1]*0.35], p.id, p.kept ? "#2a78d6" : "#6b7480"); });
  }
  if (k === "keep") {
    dots((i) => !H[i] ? null : PL[i] < 0 ? "#d6dade" : PL[i] < NK ? col(PL[i]) : "#b9bec5");
    D.planes.forEach((p, i) => {
      seg(p.p0, p.p1, p.kept ? col(i) : "#8b9097", hover && hover.kind === "plane" && hover.i === i ? 5 : 3, p.kept ? null : [5, 4]);
      if (p.kept) tick(p.p0, p.p1, p.n_room, col(i));
      p.openings.forEach((o) => { seg(o.p0, o.p1, "#16191d", 7); });
      label([(p.p0[0]+p.p1[0])/2 - p.n_room[0]*0.3, (p.p0[1]+p.p1[1])/2 - p.n_room[1]*0.3], p.id, p.kept ? "#16191d" : "#6b7480");
    });
  }
  // scale bar
  const r2 = cv.getBoundingClientRect(); ctx.strokeStyle = "#454c55"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(16, r2.height - 18); ctx.lineTo(16 + V.k, r2.height - 18); ctx.stroke();
  ctx.fillStyle = "#454c55"; ctx.font = "12px sans-serif"; ctx.textAlign = "left"; ctx.fillText("1 m", 22 + V.k, r2.height - 14);
}

// ---------- side panel
function gridCanvas(l) {
  const nc = l.solid.length, nz = l.solid[0].length, cs = 7, cvg = document.createElement("canvas");
  cvg.width = nc * cs; cvg.height = nz * cs; const g = cvg.getContext("2d");
  for (let i = 0; i < nc; i++) for (let j = 0; j < nz; j++) {
    const solid = l.solid[i][j], free = l.free[i][j];
    g.fillStyle = solid ? "#2a78d6" : free ? "#1baf7a" : "#eef1f4"; g.fillRect(i * cs, (nz - 1 - j) * cs, cs - 1, cs - 1);
  }
  for (let i = 0; i < nc; i++) if (l.code[i] === "O") { g.strokeStyle = "#e34948"; g.lineWidth = 2; g.strokeRect(i * cs + 1, 1, cs - 3, nz * cs - 3); }
  return cvg;
}
function side() {
  const s = STEPS[step];
  if (s.key !== "cut") {
    $("side").innerHTML = s.h + `<div class="lg">${s.lg.map(([c2, t]) => `<div><span class="chip" style="background:${c2}"></span>${t}</div>`).join("")}</div>`;
    return;
  }
  const l = D.lines[selLine], cuts = (l.code.match(/O+/g) || []).length;
  const pieces = D.planes.filter((p) => p.line === selLine);
  $("side").innerHTML = `<h3>④ 用射线把墙线切开</h3>
   <p>沿墙线画 ${pr.cell_m*100} cm 的网格（横轴沿墙，纵轴是高度 ${f1(l.z_lo)}–2.7 m），每格数两种射线：<b>停在这一格</b>的（这里有东西）和<b>从这一格穿过去、终点在后面</b>的（这里是空的）。一格至少 ${pr.min_rays} 条才算数。</p>
   <p>某一整列从下到上都是“空的”、没有任何“有东西”的格子、也没有墙点，就从这里切断（红框）。切开后太小的段（< ${pr.min_local_m2} m²）直接扔掉（灰色虚线）。</p>
   <p>选一条墙线看它的网格：</p>
   <div class="ln">${D.lines.map((x, i) => `<button data-l="${i}" class="${i === selLine ? "on" : ""}">${i}${x.code.includes("O") ? " ✂" : ""}</button>`).join("")}</div>
   <div class="kv"><span>墙线</span><span>${selLine}，由 ${l.locals.length} 块小墙面连成，长 ${f2(l.hi - l.lo)} m</span>
     <span>切口</span><span>${cuts} 处</span><span>切出的墙面</span><span>${pieces.map((p) => p.id).join("、") || "无（都太小）"}</span></div>
   <div class="lg"><div><span class="chip" style="background:#2a78d6"></span>有射线停在这里（有东西）</div><div><span class="chip" style="background:#1baf7a"></span>射线穿过这里（空的）</div>
     <div><span class="chip" style="background:#eef1f4;border:1px solid #dfe3e8"></span>射线不够，看不出</div><div><span class="chip" style="border:2px solid #e34948"></span>整列是空的：在这里切开</div></div>
   <div class="grid" id="gridbox"></div>`;
  $("gridbox").appendChild(gridCanvas(l));
  $("side").querySelector(".ln").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { selLine = +b.dataset.l; render(); } });
}
function render() { document.querySelectorAll("#steps button").forEach((b, i) => b.classList.toggle("on", i === step)); side(); draw(); }

// ---------- interaction
function nearest(x, y) {
  const k = STEPS[step].key, tol = 8 / V.k; let best = null;
  const dseg = (a, b) => { const ux = b[0]-a[0], uy = b[1]-a[1], L2 = ux*ux + uy*uy || 1e-9, t = Math.max(0, Math.min(1, ((x-a[0])*ux + (y-a[1])*uy) / L2));
    return Math.hypot(x - a[0] - t*ux, y - a[1] - t*uy); };
  const consider = (kind, i, a, b) => { const d = dseg(a, b); if (d < tol && (!best || d < best.d)) best = { kind, i, d }; };
  if (k === "local") D.locals.forEach((l) => consider("local", l.k, l.p0, l.p1));
  if (k === "lines" || k === "cut") D.lines.forEach((l, i) => consider("line", i, l.p0, l.p1));
  if (k === "keep") D.planes.forEach((p, i) => consider("plane", i, p.p0, p.p1));
  return best;
}
function tipText(h) {
  if (!h) return "";
  if (h.kind === "local") { const l = D.locals[h.i]; return `小墙面 #${l.k}${l.sweep ? "（扫尾找到的）" : ""}<br>来自帧 ${l.frame} 的 mask：${l.where}<br>${l.points} 个点 ≈ ${f2(l.points * D.voxel_m * D.voxel_m)} m²，并入墙线 ${l.line}`; }
  if (h.kind === "line") { const l = D.lines[h.i]; return `墙线 ${h.i}：${l.locals.length} 块小墙面，长 ${f2(l.hi - l.lo)} m${l.code.includes("O") ? "，有切口（点 ④ 看）" : ""}`; }
  if (h.kind === "plane") { const p = D.planes[h.i]; return `${p.id}${p.kept ? "（保留）" : "（丢弃：" + p.why + "）"}<br>长 ${f2(p.length_m)} m，支持 ${f2(p.area_m2)} m²，${p.frames} 帧${p.openings.length ? "<br>开口 " + p.openings.map((o) => f2(o.w) + " × " + f2(o.h) + " m").join("、") : ""}`; }
}
let drag = null;
cv.addEventListener("mousedown", (e) => { drag = [e.clientX, e.clientY, V.ox, V.oy]; cv.style.cursor = "grabbing"; });
window.addEventListener("mouseup", () => { drag = null; cv.style.cursor = "grab"; });
cv.addEventListener("mousemove", (e) => {
  const r = cv.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (drag) { V.ox = drag[2] - (e.clientX - drag[0]) / V.k; V.oy = drag[3] + (e.clientY - drag[1]) / V.k; draw(); return; }
  const h = nearest(...W2(sx, sy)), changed = JSON.stringify(h && [h.kind, h.i]) !== JSON.stringify(hover && [hover.kind, hover.i]);
  hover = h; const t = $("tip"); if (h) { t.innerHTML = tipText(h); t.style.display = "block"; t.style.left = Math.min(sx + 14, r.width - 330) + "px"; t.style.top = (sy + 14) + "px"; } else t.style.display = "none";
  if (changed) draw();
});
cv.addEventListener("click", (e) => { if (STEPS[step].key === "cut" && hover && hover.kind === "line") { selLine = hover.i; render(); } });
cv.addEventListener("mouseleave", () => { $("tip").style.display = "none"; hover = null; draw(); });
cv.addEventListener("wheel", (e) => { e.preventDefault(); const r = cv.getBoundingClientRect(), [wx, wy] = W2(e.clientX - r.left, e.clientY - r.top), f = Math.exp(-e.deltaY * 0.0015);
  V.k *= f; V.ox = wx - (wx - V.ox) / f; V.oy = wy - (wy - V.oy) / f; draw(); }, { passive: false });
cv.addEventListener("dblclick", () => { fit(); draw(); });
window.addEventListener("resize", draw);
const hs = new URLSearchParams(location.hash.slice(1)); if (hs.has("step")) step = Math.max(0, Math.min(4, +hs.get("step") - 1)); if (hs.has("line")) selLine = +hs.get("line");
fit(); render();
})();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="struct_v4_win5")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    build(a.run, a.out)


if __name__ == "__main__":
    main()
