/* Structure generation, stage 1: the white model.
 *
 * One WebGL canvas, no library, so the page opens over file://. The mesh is the
 * exported USD flattened to non-indexed triangles (world space, Z up, metres),
 * one flat colour per segment. Orbit: left drag, pan: right drag, zoom: wheel.
 */
(function () {
  "use strict";

  const DATA = window.STRUCTUREGEN || { segments: [], views: [], mesh: null };
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

  /* ---------------------------------------------------------------- decode */

  function decode(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  /* ---------------------------------------------------------------- matrix */

  const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalise = (v) => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; };

  function perspective(fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }

  function lookAt(eye, target, up) {
    const f = normalise(subtract(target, eye));
    const s = normalise(cross(f, up));
    const u = cross(s, f);
    return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
      -dot(s, eye), -dot(u, eye), dot(f, eye), 1]);
  }

  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c += 1) {
      for (let r = 0; r < 4; r += 1) {
        out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  /* ---------------------------------------------------------------- viewer */

  const FOV = Math.PI / 4;
  const UP = [0, 0, 1];
  const VERTEX = `
    attribute vec3 a_position;
    attribute vec3 a_normal;
    attribute vec3 a_colour;
    uniform mat4 u_matrix;
    varying vec3 v_colour;
    void main() {
      float light = 0.36 + 0.64 * abs(dot(normalize(a_normal), normalize(vec3(0.42, -0.5, 0.76))));
      v_colour = a_colour * light;
      gl_Position = u_matrix * vec4(a_position, 1.0);
    }`;
  const FRAGMENT = `precision mediump float; varying vec3 v_colour;
    void main() { gl_FragColor = vec4(v_colour, 1.0); }`;

  function program(gl) {
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  function createViewer(canvas, labels, mesh) {
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) {
      canvas.parentElement.insertAdjacentHTML("beforeend",
        '<p class="no-webgl">这个浏览器没有 WebGL，无法显示三维视图；下方的关键帧叠图与整体视图不受影响。</p>');
      return null;
    }
    const prog = program(gl);
    const loc = {
      position: gl.getAttribLocation(prog, "a_position"),
      normal: gl.getAttribLocation(prog, "a_normal"),
      colour: gl.getAttribLocation(prog, "a_colour"),
      matrix: gl.getUniformLocation(prog, "u_matrix"),
    };
    const position = decode(mesh.position), normal = decode(mesh.normal), colour = decode(mesh.colour);
    const count = mesh.count;
    const positionBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuf); gl.bufferData(gl.ARRAY_BUFFER, position, gl.STATIC_DRAW);
    const normalBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuf); gl.bufferData(gl.ARRAY_BUFFER, normal, gl.STATIC_DRAW);
    const colourBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colourBuf); gl.bufferData(gl.ARRAY_BUFFER, colour, gl.STATIC_DRAW);

    // Ground grid on z = 0, where the floor is.
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < position.length; i += 3) {
      for (let k = 0; k < 3; k += 1) { lo[k] = Math.min(lo[k], position[i + k]); hi[k] = Math.max(hi[k], position[i + k]); }
    }
    const centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0];
    const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2]);
    const lines = [];
    const half = Math.ceil((size / 2) / 1.0);
    for (let i = -half; i <= half; i += 1) {
      const major = i % 5 === 0;
      const c = major ? [0.72, 0.76, 0.8] : [0.85, 0.88, 0.91];
      lines.push([centre[0] + i, centre[1] - half, 0, centre[0] + i, centre[1] + half, 0], c);
      lines.push([centre[0] - half, centre[1] + i, 0, centre[0] + half, centre[1] + i, 0], c);
    }
    const gridData = new Float32Array(lines.length / 2 * 6), gridColour = new Float32Array(lines.length / 2 * 6);
    for (let i = 0; i < lines.length; i += 2) {
      const seg = lines[i], c = lines[i + 1];
      gridData.set(seg, i * 3);
      gridColour.set([c[0], c[1], c[2], c[0], c[1], c[2]], i * 3);
    }
    const gridBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf); gl.bufferData(gl.ARRAY_BUFFER, gridData, gl.STATIC_DRAW);
    const gridColourBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, gridColourBuf); gl.bufferData(gl.ARRAY_BUFFER, gridColour, gl.STATIC_DRAW);

    labels.innerHTML = "";
    for (let i = 0; i < 3; i += 1) labels.append(document.createElement("span"));
    const camera = { yaw: -0.95, pitch: 0.55, distance: size * 1.75, target: centre };
    let pending = false;

    function eye() {
      const cp = Math.cos(camera.pitch);
      return [camera.target[0] + camera.distance * cp * Math.cos(camera.yaw),
              camera.target[1] + camera.distance * cp * Math.sin(camera.yaw),
              camera.target[2] + camera.distance * Math.sin(camera.pitch)];
    }

    function bind(buffer, location, sizeAttr) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, sizeAttr, gl.FLOAT, false, 0, 0);
    }

    function draw() {
      pending = false;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.93, 0.95, 0.96, 1);
      gl.enable(gl.DEPTH_TEST);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const view = lookAt(eye(), camera.target, UP);
      const proj = perspective(FOV, w / h, Math.max(0.05, size * 0.01), size * 20);
      const mvp = multiply(proj, view);
      gl.useProgram(prog);
      gl.uniformMatrix4fv(loc.matrix, false, mvp);
      bind(gridBuf, loc.position, 3); bind(gridColourBuf, loc.colour, 3);
      gl.disableVertexAttribArray(loc.normal);
      gl.vertexAttrib3f(loc.normal, 0, 0, 1);
      gl.drawArrays(gl.LINES, 0, gridData.length / 3);
      bind(positionBuf, loc.position, 3); bind(normalBuf, loc.normal, 3); bind(colourBuf, loc.colour, 3);
      gl.drawArrays(gl.TRIANGLES, 0, count);
      placeLabels(mvp, w, h);
    }

    function placeLabels(mvp, w, h) {
      const origin = [lo[0], lo[1], 0];
      const axes = [[size * 0.9, 0, 0, "X", "x"], [0, size * 0.9, 0, "Y", "y"], [0, 0, size * 0.9, "Z", "z"]];
      axes.forEach(([dx, dy, dz, name, cls], i) => {
        const p = [origin[0] + dx, origin[1] + dy, origin[2] + dz, 1];
        const x = mvp[0] * p[0] + mvp[4] * p[1] + mvp[8] * p[2] + mvp[12];
        const y = mvp[1] * p[0] + mvp[5] * p[1] + mvp[9] * p[2] + mvp[13];
        const ww = mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15];
        const span = labels.children[i];
        if (ww <= 0) { span.hidden = true; return; }
        span.hidden = false;
        span.className = `axis-label axis-${cls}`;
        span.textContent = name;
        span.style.left = `${(x / ww * 0.5 + 0.5) * canvas.clientWidth}px`;
        span.style.top = `${(-y / ww * 0.5 + 0.5) * canvas.clientHeight}px`;
      });
    }

    function requestDraw() {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(draw);
    }

    let drag = null;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 }; canvas.classList.add("dragging"); });
    window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("dragging"); });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const forward = normalise(subtract(camera.target, eye()));
        const right = normalise(cross(forward, UP));
        const up = cross(right, forward);
        const scale = (camera.distance * 1.2) / Math.max(1, canvas.clientHeight);
        for (let i = 0; i < 3; i += 1) camera.target[i] += (-dx * right[i] + dy * up[i]) * scale;
      } else {
        camera.yaw -= dx * 0.008;
        camera.pitch = Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.008));
      }
      requestDraw();
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      camera.distance = Math.max(size * 0.3, Math.min(size * 12, camera.distance * Math.exp(e.deltaY * 0.0012)));
      requestDraw();
    }, { passive: false });
    window.addEventListener("resize", requestDraw);
    requestDraw();
    return { requestDraw };
  }

  /* ------------------------------------------------------------------ page */

  $("segCount").textContent = DATA.segments.length;
  const total = DATA.segments.reduce((sum, s) => sum + (s.volume_m3 || 0), 0);
  $("wallVolume").textContent = `${total.toFixed(2)} m³`;
  $("frameCount").textContent = DATA.frames.length;

  const stages = [
    ["stage1", "1 白模生成", "脚本", false],
    ["stage2", "2 agent 的输入", DATA.evidence ? "任务卡 · 俯视图 · 第一面墙" : "未开始", !DATA.evidence],
    ["stage3", "3 外观 agent", "设计稿", true],
    ["stage3run", "3 运行结果", DATA.agent ? `${DATA.agent.replies} 次回复 · $${DATA.agent.usd.toFixed(2)}` : "未运行", !DATA.agent],
    ["stage4", "4 导出与校验", "未开始", true],
  ];
  $("timeline").innerHTML = stages.map(([id, title, tag, pending]) =>
    `<a href="#${id}" class="${pending ? "pending" : "active"}"><b>${title}</b>${tag}</a>`).join("");

  $("legend").innerHTML = DATA.segments.map((s) =>
    `<li><span class="swatch" style="background:${s.colour}"></span>${s.id}<span class="role">${s.role === "exterior" ? "外墙" : "内墙"}</span></li>`).join("");
  $("viewerNote").textContent = [
    DATA.height_m ? `层高 ${DATA.height_m} m（覆盖值）` : "层高取交接假定值 2.80 m",
    `切缝：${Object.entries(DATA.cuts).filter(([, v]) => v.length).map(([k, v]) => `${k} @ ${v.map((x) => x.toFixed(3)).join(" / ")}`).join("；") || "无"}`,
  ].join("。");

  const mismatches = [];
  $("segmentRows").innerHTML = DATA.segments.map((s) => {
    const error = s.volume_m3 === null ? null : s.volume_m3 - s.expected_volume_m3;
    if (error !== null && Math.abs(error) > 1e-3) mismatches.push(s.id);
    return `<tr>
      <td><span class="swatch" style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${s.colour};margin-right:6px"></span>${s.id}</td>
      <td>${s.wall}</td>
      <td class="n">${fmt(s.a0, 3)} – ${fmt(s.a1, 3)}</td>
      <td class="n">${fmt(s.length_m, 3)}</td>
      <td class="n">${fmt(s.thickness_m, 2)}</td>
      <td class="n">${s.openings || "—"}</td>
      <td class="n">${s.faces ?? "—"}</td>
      <td class="${s.closed ? "yes" : "no"}">${s.closed ? "是" : "否"}</td>
      <td class="n">${fmt(s.volume_m3, 4)}</td>
      <td class="n">${error === null ? "—" : (error >= 0 ? "+" : "") + error.toFixed(4)}</td>
    </tr>`;
  }).join("");
  $("tableNote").textContent = mismatches.length
    ? `体积与解析值不符：${mismatches.join("、")}`
    : "每段体积与「占地 × 高 − 洞口」逐段相等（误差 < 0.001 m³）。解析值由 segments.py 计算，体积由 Blender 与 pxr 各量一次。";

  $("views").innerHTML = DATA.views.map((v) => `<figure>
      <div class="stack">
        <img src="${v.photo}" alt="关键帧 ${v.fid} 照片" loading="lazy">
        <img class="model" src="${v.model}" alt="关键帧 ${v.fid} 白模" loading="lazy">
        <img class="edges" src="${v.edges}" alt="关键帧 ${v.fid} 白模轮廓" loading="lazy">
      </div>
      <figcaption><b>帧 ${v.fid}</b>${v.walls.map((w) => `<span class="chip">${w}</span>`).join("")}</figcaption>
    </figure>`).join("");
  $("viewCount").textContent = `· ${DATA.views.length} 帧，点击展开`;
  const opacity = $("opacity");
  const applyOpacity = () => {
    document.documentElement.style.setProperty("--model-opacity", opacity.value / 100);
    $("opacityValue").textContent = `${opacity.value}%`;
  };
  opacity.addEventListener("input", applyOpacity);
  applyOpacity();
  $("edgesOnly").addEventListener("change", (e) => {
    document.querySelectorAll(".stack").forEach((el) => el.classList.toggle("edges", e.target.checked));
  });


  /* --------------------------------------------------------------- stage 2 */

  const EV = DATA.evidence;
  if (EV) renderInputs(EV);

  function renderInputs(ev) {
    const card = ev.card, facts = card.scene_facts;
    const colours = Object.fromEntries(DATA.segments.map((s) => [s.id, s.colour]));
    const xy = (p) => `(${p[0].toFixed(3)}, ${p[1].toFixed(3)})`;
    const dot = (s) => `<span class="dot" style="background:${colours[s.id]}">${s.number}</span>`;
    const first = card.segments.find((s) => s.id === card.first.segment);

    $("inputList").innerHTML = [
      ["任务卡", "开局", "文字（Markdown 表格）", "场景、Blender 命名、墙段几何与法向、洞口、每段关键帧、假定值", "交接 + 白模几何"],
      ["俯视图", "开局", "1 张图", "每段一种颜色，标序号、法向、洞口", "白模几何"],
      ["第一面墙的图", "开局", `${ev.first_pairs.length * 2} 张图`, `#${first.number} ${first.id} 的 _rgb.jpg 与 _boxed.jpg`, "交接原文件"],
      ["后续各段的图", "逐段", "每段 2–6 张图", "同上，做完一段给下一段", "交接原文件"],
      ["工具", "全程", "MCP + 自定义", "<code>compare</code>、<code>look</code>、<code>get_scene_info</code>、<code>declare</code>、<code>execute_blender_code</code>、<code>finish_segment</code>、<code>commit</code>", "—"],
    ].map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");

    $("cardSize").textContent = `task_card.md · ${ev.card_chars.toLocaleString()} 字`;
    $("sceneFacts").innerHTML = [
      ["坐标系", `${facts.frame}`], ["单位 / 向上", `${facts.units} / +${facts.up_axis}`], ["地面", `z = ${facts.floor_z}`],
      ["墙段", `${facts.segments}（外墙 ${facts.exterior}、内墙 ${facts.interior}）`], ["洞口", facts.openings], ["关键帧", facts.key_frames],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    $("blenderNames").innerHTML = [
      ["物体", `<code>${card.blender.object}</code>`], ["材质槽", `<code>${card.blender.material_slot}</code>`],
      ["面", card.blender.faces.map((f) => `<code>${f}</code>`).join(" ")], ["front", "房间一侧"], ["reveal", "洞口侧壁"],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    $("assumedRows").innerHTML = card.assumed.map((a) => `<tr><td>${a.item}</td><td>${a.value}</td><td>${a.kind}</td><td>${a.note}</td></tr>`).join("");
    $("excluded").innerHTML = `交接不含：${card.not_included.map((x) => `<span>${x}</span>`).join("")}`;

    $("cardSegments").innerHTML = card.segments.map((s) => `<tr class="${s.id === first.id ? "first" : ""}">
      <td>${dot(s)}</td><td>${s.id}</td><td>${s.wall}</td><td>${s.role === "exterior" ? "外墙" : "内墙"}</td>
      <td class="n">${xy(s.front[0])}</td><td class="n">${xy(s.front[1])}</td>
      <td class="n">${s.along_m[0].toFixed(3)}–${s.along_m[1].toFixed(3)}</td><td class="n">${s.length_m.toFixed(3)}</td>
      <td class="n">${s.thickness_m}</td><td class="n">${s.height_m}</td>
      <td class="n">(${s.normal_room[0].toFixed(3)}, ${s.normal_room[1].toFixed(3)}, 0)</td>
      <td>${s.openings.map((o) => `${o.from_m.toFixed(3)}–${o.to_m.toFixed(3)} × ${o.z_m[0]}–${o.z_m[1]} m`).join("；") || "—"}</td></tr>`).join("");

    $("frameRows").innerHTML = card.segments.map((s) => (s.frames.length ? s.frames : [null]).map((f, i) => `<tr class="${i === 0 ? "group-start" : ""} ${s.id === first.id ? "first" : ""}">
      <td>${i === 0 ? dot(s) : ""}</td><td>${i === 0 ? s.id : ""}</td>
      ${f ? `<td>${f.frame}</td><td>${f.role}</td><td class="n">${f.covers_m[0]}–${f.covers_m[1]}</td>
      <td class="n">${f.view_angle_deg}</td><td class="n">${f.reprojection_px ?? "—"}</td><td class="n">${f.visible_area_m2}</td>
      <td class="n">${f.visible_pixels.toLocaleString()}</td><td class="n">${f.clarity ?? "—"}</td>` : `<td colspan="8">—</td>`}</tr>`).join("")).join("");
    $("cardText").textContent = ev.card_text;

    $("planImage").src = ev.plan;

    const cov = card.first.coverage;
    $("firstLabel").textContent = `#${first.number} ${first.id}`;
    $("firstFacts").innerHTML = [
      ["墙段", `${dot(first)} ${first.id}（${first.wall}，${first.role === "exterior" ? "外墙" : "内墙"}）`],
      ["front 起点 → 终点", `${xy(first.front[0])} → ${xy(first.front[1])}`],
      ["沿墙 / 长 / 厚 / 高", `${first.along_m[0].toFixed(3)}–${first.along_m[1].toFixed(3)} m / ${first.length_m.toFixed(3)} m / ${first.thickness_m} m / ${first.height_m} m`],
      ["front 法向", `(${first.normal_room[0].toFixed(3)}, ${first.normal_room[1].toFixed(3)}, 0)`],
      ["洞口", first.openings.length ? first.openings.map((o) => `${o.from_m}–${o.to_m} × ${o.z_m[0]}–${o.z_m[1]} m`).join("；") : "无（交接未测到）"],
      ["mask 覆盖", `${cov.seen_m2.toFixed(2)} m² / 墙面 ${cov.face_m2.toFixed(2)} m²（${(cov.seen_share * 100).toFixed(0)}%），${Object.keys(cov.per_frame_m2).length} 帧取并集`],
      ["选它的原因", "外墙中 mask 覆盖面积最大（见下方排序）"],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    $("firstFrames").innerHTML = card.first.frames.map((f) => `<tr><td>${f.frame}</td><td>${f.role}</td>
      <td class="n">${f.covers_m[0]}–${f.covers_m[1]}</td><td class="n">${f.mask_area_m2.toFixed(2)}</td>
      <td class="n">${f.view_angle_deg}</td><td class="n">${f.reprojection_px ?? "—"}</td>
      <td><code>first/${String(f.frame).padStart(4, "0")}_rgb.jpg</code> <code>first/${String(f.frame).padStart(4, "0")}_boxed.jpg</code></td></tr>`).join("");
    $("firstImages").innerHTML = ev.first_pairs.map((p) => `
      <figure><img src="${p.rgb}" alt="帧 ${p.frame} 原图" loading="lazy"><figcaption><b>帧 ${p.frame}</b> · _rgb.jpg</figcaption></figure>
      <figure><img src="${p.boxed}" alt="帧 ${p.frame} 墙体框" loading="lazy"><figcaption><b>帧 ${p.frame}</b> · _boxed.jpg</figcaption></figure>`).join("");

    const chosen = ev.ranking.chosen;
    $("rankRows").innerHTML = ev.ranking.ranking.map((r) => `<tr class="${r.segment === chosen ? "chosen" : ""} ${r.role}">
      <td>${r.rank}</td><td>${r.number}</td><td>${r.segment}</td><td>${r.role === "exterior" ? "外墙" : "内墙"}</td>
      <td class="n">${r.frames}</td><td class="n">${r.seen_m2.toFixed(2)} m²</td><td class="n">${(r.seen_share * 100).toFixed(0)}%</td>
      <td>${r.best_frame ?? "—"}</td><td class="n">${r.best_view_angle_deg ?? "—"}°</td><td class="n">${r.best_reprojection_px ?? "—"} px</td></tr>`).join("");
  }

  /* --------------------------------------------------------------- stage 3 run */

  const AG = DATA.agent;
  if (AG) renderRun(AG);
  else $("stage3run").remove();

  function renderRun(ag) {
    const esc = (t) => String(t ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    const rgbCss = (c) => (c && c.length ? `rgb(${c[0]},${c[1]},${c[2]})` : "transparent");
    const rgbTxt = (c) => (c && c.length ? c.join(", ") : "—");
    const segColour = Object.fromEntries(DATA.segments.map((s) => [s.id, s.colour]));
    const dot = (w) => w.number ? `<span class="dot" style="background:${segColour[w.segment]}">${w.number}</span>` : "";
    const walls = ag.walls.filter((w) => w.number && !w.floor);
    const wrap = ag.walls.find((w) => w.segment === "wrapup");
    const KIND = { skirting: "踢脚线", rail: "腰线", casing: "门套" };

    $("runModel").textContent = ag.served_by || ag.model;
    $("runDir").textContent = ag.dir.split("/").slice(-2).join("/");
    const floorsDone = ag.walls.filter((w) => w.floor);
    const srcNote = ag.sources.length > 1
      ? `分两次跑：${ag.sources.map((x) => `${x.what === "this run" ? "地板 + 收尾" : "墙（之前的运行，回放到加了地板的白模上）"} ${x.replies} 次回复、$${x.usd.toFixed(2)}`).join("；")}。`
      : "";
    $("runLede").innerHTML = `${floorsDone.length ? `${floorsDone.length} 块地板、` : ""}${walls.length} 面墙：逐个 declare → 改 → 自动检查与对比图 → finish_segment，最后收尾补色并 commit。
      ${srcNote}状态 <b>${ag.status}</b>；${walls.filter((w) => w.status === "done").length} 面墙完成，${walls.filter((w) => w.status !== "done").length} 面未收敛。
      下方所有颜色与尺寸读自 <code>appearance.blend</code>。`;
    if (floorsDone.length) {
      $("floorRows").innerHTML = floorsDone.map((f) => {
        const top = f.faces.top || {};
        const tl = top.tiles;
        const d = f.declared || {};
        return `<tr><td>${f.segment}<span class="rgbtxt">${f.kind === "room" ? "房间" : "门槛"} · ${f.area_m2.toFixed(2)} m²</span></td>
          <td class="n">${f.replies}</td><td class="n">$${f.usd.toFixed(2)}</td>
          <td class="col"><span class="cpair"><i style="background:${rgbCss(top.rgb)}"></i>${tl ? `<i style="background:${rgbCss(tl.joint_rgb)}"></i>` : ""}</span>
            <span class="rgbtxt">${tl ? "砖" : "色"} ${rgbTxt(top.rgb)}</span>${tl ? `<span class="rgbtxt">缝 ${rgbTxt(tl.joint_rgb)}</span>` : ""}</td>
          <td>${tl ? `瓷砖/板 ${tl.size_m[0].toFixed(2)}×${tl.size_m[1].toFixed(2)} m · 缝 ${(tl.joint_m * 1000).toFixed(0)} mm · 转 ${tl.angle_deg.toFixed(1)}°` : "纯色"}
            <span class="rgbtxt">依据帧 ${(d.surface?.basis || []).join(", ") || "—"}</span></td>
          <td class="wide">${esc(d.surface?.material)}</td><td class="wide">${esc(d.notes || f.summary)}</td></tr>`;
      }).join("");
    } else {
      $("floorBlock").remove();
    }

    const t = ag.tokens;
    const tiles = [
      ["状态", ag.status, "commit 通过"],
      ["模型回复", ag.replies, `上限 每面墙 15`],
      ["费用", `$${ag.usd.toFixed(2)}`, "保护线 $20"],
      ["用时", `${Math.floor(ag.seconds / 60)} 分 ${ag.seconds % 60} 秒`, `平均每次回复 ${(ag.usage.reduce((a, r) => a + r.seconds, 0) / ag.usage.length).toFixed(1)} 秒`],
      ["工具调用", ag.tool_calls, Object.entries(ag.tools).map(([k, v]) => `${k.replace("execute_blender_code", "execute")} ${v}`).join(" · ")],
      ["输入 token", t.input_tokens.toLocaleString(), `单次最多 ${t.max_input.toLocaleString()}`],
      ["缓存命中", `${(t.cached_tokens / t.input_tokens * 100).toFixed(1)}%`, `${t.cached_tokens.toLocaleString()} token`],
      ["输出 token", t.output_tokens.toLocaleString(), `其中推理 ${t.reasoning_tokens.toLocaleString()}`],
    ];
    $("runTiles").innerHTML = tiles.map(([k, v, n]) => `<div><span>${k}</span><b>${v}</b><small>${n}</small></div>`).join("");

    const order = ag.walls.map((w) => w.segment);
    const wallTint = (w) => segColour[w] || (w.startsWith("room_") ? "#c8b892" : w.startsWith("threshold") ? "#a8987a" : "#8a8f98");
    const maxCost = Math.max(...ag.usage.map((r) => r.cost));
    const letter = { declare: "D", execute_blender_code: "E", compare: "C", finish_segment: "F", commit: "M" };
    $("replyBars").innerHTML = ag.usage.map((r) => `<div class="bar" title="#${r.n} ${r.wall} · $${r.cost.toFixed(3)} · 输入 ${r.input.toLocaleString()}（缓存 ${r.cached.toLocaleString()}）· 输出 ${r.output} · ${r.calls.join(", ")}">
      <em>${r.cost.toFixed(2)}</em><i style="height:${(r.cost / maxCost * 120).toFixed(0)}px;background:${wallTint(r.wall)}"></i></div>`).join("");
    $("replyBars").insertAdjacentHTML("afterend", `<div class="reply-calls">${ag.usage.map((r) => `<span>${r.calls.map((c) => letter[c] || "?").join("")}</span>`).join("")}</div>`);
    $("replyLegend").innerHTML = order.map((w) => {
      const x = ag.walls.find((v) => v.segment === w);
      return `<span><i style="background:${wallTint(w)}"></i>${w} · ${x.replies} 次 · $${x.usd.toFixed(2)}</span>`;
    }).join("");

    // R2 viewer
    $("finalObjects").textContent = walls.length + walls.reduce((a, w) => a + w.trims.length, 0);
    const swatches = new Map();
    walls.forEach((w) => {
      const f = w.faces.front;
      if (f) swatches.set(rgbTxt(f.rgb), [f.rgb, (swatches.get(rgbTxt(f.rgb))?.[1] || []).concat(w.segment)]);
      w.trims.forEach((tr) => swatches.set(rgbTxt(tr.rgb), [tr.rgb, (swatches.get(rgbTxt(tr.rgb))?.[1] || []).concat(`${w.segment} ${KIND[tr.kind]}`)]));
    });
    $("finalLegend").innerHTML = [...swatches.values()].map(([c, who]) =>
      `<li><span class="swatch" style="background:${rgbCss(c)}"></span><span>${rgbTxt(c)}<br><small class="role" style="margin:0">${who.join("、")}</small></span></li>`).join("");
    createViewer($("finalViewer"), $("finalLabels"), ag.mesh);

    // R3 table
    const trimText = (tr) => {
      const size = tr.kind === "casing"
        ? `面宽 ${(tr.width * 100).toFixed(1)} cm · ${tr.sides === "FRONT" ? "仅房间一侧" : tr.sides === "BACK" ? "仅背面" : "两面"}`
        : `z ${tr.z[0].toFixed(2)}–${tr.z[1].toFixed(2)} m · 沿墙 ${tr.along[0].toFixed(2)}–${tr.along[1].toFixed(2)} m`;
      return `<span class="trim"><span class="sq" style="background:${rgbCss(tr.rgb)}"></span><b>${KIND[tr.kind]}</b> ${size} · 凸出 ${(tr.depth * 1000).toFixed(0)} mm<span class="rgbtxt">${rgbTxt(tr.rgb)}</span></span>`;
    };
    $("wallRows").innerHTML = walls.map((w) => {
      const f = w.faces.front || {};
      const d = w.declared || {};
      return `<tr>
        <td>${dot(w)}</td><td>${w.segment}<span class="rgbtxt">${w.role === "exterior" ? "外墙" : "内墙"} · ${w.length_m.toFixed(2)} m</span></td>
        <td class="n">${w.replies}</td><td class="n">$${w.usd.toFixed(2)}</td><td class="n">${w.edits}</td>
        <td class="col"><span class="cpair" title="左 agent / 右 照片中位数"><i style="background:${rgbCss(f.rgb)}"></i><i style="background:${rgbCss(w.photo_rgb)}"></i></span>
          <span class="rgbtxt">agent ${rgbTxt(f.rgb)}</span><span class="rgbtxt">照片 ${rgbTxt(w.photo_rgb)}</span></td>
        <td class="wide">${esc(d.surface?.material)}<span class="rgbtxt">依据帧 ${(d.surface?.basis || []).join(", ")} · 粗糙度 ${f.roughness ?? "—"}</span></td>
        <td class="wide">${w.trims.length ? w.trims.map(trimText).join("") : "—"}</td>
        <td class="wide">${esc(d.notes)}</td></tr>`;
    }).join("");

    // R4 final compare
    $("finalCount").textContent = `${ag.final.length} 帧`;
    $("finalGrid").innerHTML = ag.final.map((f) => `<figure><img src="${f.image}" alt="帧 ${f.frame} 照片与最终模型" loading="lazy">
      <figcaption><b>帧 ${f.frame}</b>${f.walls.map((w) => `<span class="chip">${w}</span>`).join("")}</figcaption></figure>`).join("");

    // R5 face matrix
    $("faceHead").innerHTML = `<tr><th class="face">段</th>${ag.faces.map((f) => `<th>${f}</th>`).join("")}</tr>`;
    $("faceRows").innerHTML = walls.map((w) => `<tr><td class="face">${dot(w)} ${w.segment}</td>${ag.faces.map((f) => {
      const x = w.faces[f];
      if (!x) return `<td class="cell empty">—</td>`;
      return `<td class="cell" title="${rgbTxt(x.rgb)}"><span style="background:${rgbCss(x.rgb)}"></span>${x.item}</td>`;
    }).join("")}</tr>`).join("");

    // R6 per-wall steps
    const TOOL_CN = { brief: "开场材料（harness 给）", declare: "declare 记录所见", execute_blender_code: "execute_blender_code 改模型",
      compare: "compare 看对比图", finish_segment: "finish_segment 结束这面墙", commit: "commit 全场景检查" };
    const argsView = (st) => {
      if (st.tool === "execute_blender_code") return `<div class="reply-text">${esc(st.args.description)}</div><pre>${esc(st.args.code)}</pre>`;
      if (st.tool === "declare") {
        const fd = st.args.findings || {};
        const rows = [["墙面", `${esc(fd.surface?.material)} · rgb ${rgbTxt(fd.surface?.rgb)} · 依据 ${(fd.surface?.basis || []).join(", ")}`]]
          .concat((fd.trims || []).map((tr) => [KIND[tr.kind] || tr.kind, `${tr.z ? `z ${tr.z.join("–")} m · ` : ""}rgb ${rgbTxt(tr.rgb)} · ${esc(tr.note)}`]))
          .concat([["备注", esc(fd.notes)]]);
        return `<dl class="declared">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
      }
      if (st.tool === "finish_segment") return `<div class="reply-text">${esc(st.args.summary)}</div>`;
      if (st.tool === "compare") return `<div class="reply-text">帧 ${st.args.frame}</div>`;
      return "";
    };
    const stepHtml = (st) => {
      const who = st.reply ? `模型第 ${st.reply} 次回复` : "harness";
      const flags = [st.refused ? "被拒绝" : "", st.rolled_back ? "已回滚" : "", st.trims.length ? "饰线问题" : ""].filter(Boolean);
      const left = `${argsView(st)}<pre>${esc(st.text)}</pre>`;
      const right = st.image ? `<img src="${st.image}" alt="对比图" loading="lazy">`
        : st.photos.length ? `<div class="photos">${st.photos.map((p) => `<img src="${p}" alt="照片" loading="lazy">`).join("")}</div>` : "";
      const extra = st.image && st.photos.length ? `<div class="photos" style="margin-top:6px">${st.photos.map((p) => `<img src="${p}" alt="照片" loading="lazy">`).join("")}</div>` : "";
      return `<li class="step"><div class="step-head"><span class="tool t-${st.tool}">${TOOL_CN[st.tool] || st.tool}</span>
        <span class="who">#${st.n} · ${who}</span>${flags.map((f) => `<span class="flag">${f}</span>`).join("")}</div>
        <div class="step-body ${right ? "" : "single"}"><div>${left}</div>${right ? `<div>${right}${extra}</div>` : ""}</div></li>`;
    };
    $("wallSteps").innerHTML = ag.walls.map((w, i) => {
      const title = w.floor ? `地板 ${w.segment}` : w.number ? `${dot(w)} ${w.segment}` : "收尾";
      const sub = `${w.replies} 次回复 · $${w.usd.toFixed(2)} · ${w.steps.filter((s) => s.tool !== "brief").length} 次工具调用 · ${w.status}`;
      return `<details class="fold"${i === 0 ? " open" : ""}><summary>${title} <span class="count">· ${sub}</span></summary>
        <ol class="steps">${w.steps.map(stepHtml).join("")}</ol></details>`;
    }).join("");
  }

  const canvas = $("viewer");
  if (DATA.mesh && DATA.mesh.count) createViewer(canvas, $("viewerLabels"), DATA.mesh);
  else canvas.replaceWith(Object.assign(document.createElement("p"), { textContent: "没有可显示的网格。" }));
})();
