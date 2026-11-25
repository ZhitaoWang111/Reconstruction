"use strict";

const registry = new Map();
const frameCache = new Map();
const loadedFrameScripts = new Map();
const maxFrameCache = 3;
let manifest = null;
let currentFrameIndex = 0;
const selectedInstanceIndices = new Set();
let imageMode = "rgb";
let currentImage = null;
let currentMaskBounds = [];
let loadToken = 0;
let backgroundCloud = null;
let currentCloud = null;
let glState = null;

window.registerMaskCloud = function registerMaskCloud(key, payload) {
  registry.set(String(key), payload);
};

function $(id) {
  return document.getElementById(id);
}

function setStatus(text, isError = false) {
  const node = $("status");
  node.textContent = text;
  node.style.color = isError ? "#ff6961" : "";
}

function decodeBase64ToTypedArray(base64, Type) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Type(bytes.buffer);
}

function popcount32(value) {
  value = value >>> 0;
  value = value - ((value >>> 1) & 0x55555555);
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function scriptLoad(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.assetSrc = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`无法加载 ${src}`));
    document.head.appendChild(script);
  });
}

async function ensurePayload(frame) {
  const key = String(frame.source_frame_id);
  if (!registry.has(key)) {
    await scriptLoad(frame.asset);
  }
  if (!registry.has(key)) {
    throw new Error(`asset 已加载但没有注册帧 ${key}`);
  }
  return registry.get(key);
}

async function ensureBackground() {
  if (backgroundCloud) return backgroundCloud;
  if (!registry.has("background")) {
    await scriptLoad(manifest.background_asset);
  }
  const payload = registry.get("background");
  if (!payload) throw new Error("background asset 未注册");
  backgroundCloud = {
    count: payload.count,
    positions: decodeBase64ToTypedArray(payload.positions_f32_base64, Float32Array),
    colors: decodeBase64ToTypedArray(payload.colors_u8_base64, Uint8Array),
  };
  return backgroundCloud;
}

async function loadFrame(index) {
  const token = ++loadToken;
  const frame = manifest.frames[index];
  setStatus(`加载 frame ${frame.source_frame_id} ...`);
  let decoded = frameCache.get(frame.source_frame_id);
  if (!decoded) {
    const payload = await ensurePayload(frame);
    decoded = {
      payload,
      asset: frame.asset,
      sourceFrameId: frame.source_frame_id,
      positions: decodeBase64ToTypedArray(payload.positions_f32_base64, Float32Array),
      colors: decodeBase64ToTypedArray(payload.colors_u8_base64, Uint8Array),
      maskBits: decodeBase64ToTypedArray(payload.mask_bits_u32_base64, Uint32Array),
      pixelIndices: decodeBase64ToTypedArray(payload.pixel_indices_u32_base64, Uint32Array),
    };
    frameCache.set(frame.source_frame_id, decoded);
    loadedFrameScripts.set(frame.source_frame_id, frame.asset);
    while (frameCache.size > maxFrameCache) {
      const firstKey = frameCache.keys().next().value;
      if (firstKey === frame.source_frame_id) break;
      frameCache.delete(firstKey);
      registry.delete(String(firstKey));
      const asset = loadedFrameScripts.get(firstKey);
      loadedFrameScripts.delete(firstKey);
      if (asset) {
        document.querySelectorAll(`script[data-asset-src="${cssEscape(asset)}"]`).forEach((node) => node.remove());
      }
    }
  }
  const [image] = await Promise.all([loadImage(frame.rgb_image), ensureBackground()]);
  if (token !== loadToken) return;
  currentCloud = decoded;
  currentImage = image;
  currentFrameIndex = index;
  selectedInstanceIndices.clear();
  updateFrameControls();
  draw2d(image);
  drawInstances();
  updateCloudBuffers();
  renderCloud();
  setStatus(`已加载 ${frame.source_frame_id}: ${frame.instances.length} 个原始 mask, ${frame.point_count.toLocaleString()} 个落点`);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`无法加载图像 ${src}`));
    image.src = src;
  });
}

function setupControls() {
  $("prevFrame").addEventListener("click", () => loadFrame(Math.max(0, currentFrameIndex - 1)).catch(showError));
  $("nextFrame").addEventListener("click", () => loadFrame(Math.min(manifest.frames.length - 1, currentFrameIndex + 1)).catch(showError));
  $("frameSelect").addEventListener("change", (event) => loadFrame(Number(event.target.value)).catch(showError));
  $("frameSlider").addEventListener("input", (event) => loadFrame(Number(event.target.value)).catch(showError));
  $("modeRgb").addEventListener("click", () => setImageMode("rgb"));
  $("modeAllMasks").addEventListener("click", () => setImageMode("all"));
  $("modeSelectedMasks").addEventListener("click", () => setImageMode("selected"));
  $("showBackground").addEventListener("change", () => renderCloud());
  $("colorMode").addEventListener("change", () => {
    updateCloudBuffers();
    renderCloud();
  });
  $("resetView").addEventListener("click", () => {
    glState.rotationX = -0.35;
    glState.rotationY = 0.45;
    glState.distance = 3.2;
    glState.panX = 0;
    glState.panY = 0;
    renderCloud();
  });
}

function setImageMode(mode) {
  imageMode = mode;
  if (mode === "all") selectedInstanceIndices.clear();
  $("modeRgb").classList.toggle("active", mode === "rgb");
  $("modeAllMasks").classList.toggle("active", mode === "all");
  $("modeSelectedMasks").classList.toggle("active", mode === "selected");
  updateInstanceSelection();
  if (currentImage) draw2d(currentImage);
  updateCloudBuffers();
  renderCloud();
}

function selectedMaskBits() {
  let bits = 0;
  for (const index of selectedInstanceIndices) bits |= 1 << index;
  return bits >>> 0;
}

function updateInstanceSelection() {
  const frame = manifest.frames[currentFrameIndex];
  $("instanceSummary").textContent = `已选 ${selectedInstanceIndices.size} / ${frame.instances.length}`;
  for (const button of $("instanceList").querySelectorAll("button")) {
    const selected = selectedInstanceIndices.has(Number(button.dataset.instanceIndex));
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function updateFrameControls() {
  $("frameSelect").value = String(currentFrameIndex);
  $("frameSlider").value = String(currentFrameIndex);
  const frame = manifest.frames[currentFrameIndex];
  $("frameMeta").textContent = `sequence ${frame.sequence_index} / source frame ${frame.source_frame_id} / ${frame.point_count.toLocaleString()} union points / ${frame.overlap_point_count.toLocaleString()} overlap points`;
  $("frameDownloads").innerHTML = `当前帧: <a href="${escapeAttr(frame.npz)}">NPZ</a> / <a href="${escapeAttr(frame.ply)}">PLY</a>`;
}

function draw2d(image) {
  const canvas = $("imageCanvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  currentMaskBounds = [];
  $("maskLabels").replaceChildren();
  if (!currentCloud || imageMode === "rgb") return;
  const overlay = ctx.createImageData(canvas.width, canvas.height);
  const frame = manifest.frames[currentFrameIndex];
  const selectionBits = selectedMaskBits();
  const bounds = new Map();
  for (let i = 0; i < currentCloud.pixelIndices.length; i += 1) {
    const bits = imageMode === "selected" ? (currentCloud.maskBits[i] & selectionBits) >>> 0 : currentCloud.maskBits[i];
    if (bits === 0) continue;
    const px = currentCloud.pixelIndices[i];
    const offset = px * 4;
    let color = [175, 82, 222];
    if (popcount32(bits) === 1) {
      const instanceIndex = Math.floor(Math.log2(bits));
      color = frame.instances[instanceIndex]?.color || color;
    }
    overlay.data[offset] = color[0];
    overlay.data[offset + 1] = color[1];
    overlay.data[offset + 2] = color[2];
    overlay.data[offset + 3] = 130;
    if (imageMode === "selected") {
      const x = px % canvas.width;
      const y = Math.floor(px / canvas.width);
      let remaining = bits;
      while (remaining) {
        const index = 31 - Math.clz32(remaining & -remaining);
        const box = bounds.get(index) || { index, minX: x, maxX: x, minY: y, maxY: y };
        box.minX = Math.min(box.minX, x);
        box.maxX = Math.max(box.maxX, x);
        box.minY = Math.min(box.minY, y);
        box.maxY = Math.max(box.maxY, y);
        bounds.set(index, box);
        remaining = (remaining & (remaining - 1)) >>> 0;
      }
    }
  }
  const temp = document.createElement("canvas");
  temp.width = canvas.width;
  temp.height = canvas.height;
  temp.getContext("2d").putImageData(overlay, 0, 0);
  ctx.drawImage(temp, 0, 0);
  currentMaskBounds = [...bounds.values()].sort((a, b) => a.index - b.index);
  drawMaskLabels(currentMaskBounds);
}

function drawMaskLabels(bounds) {
  const layer = $("maskLabels");
  layer.replaceChildren();
  const canvas = $("imageCanvas");
  const scaleX = layer.clientWidth / canvas.width;
  const scaleY = layer.clientHeight / canvas.height;
  const placed = [];
  for (const box of bounds) {
    const instance = manifest.frames[currentFrameIndex].instances[box.index];
    const label = document.createElement("span");
    label.className = "maskLabel";
    label.dataset.instanceIndex = String(box.index);
    label.textContent = `${instance.instance_id} · ${instance.class_name}`;
    label.style.setProperty("--mask-color", `rgb(${instance.color.join(",")})`);
    layer.appendChild(label);
    const width = label.offsetWidth;
    const height = label.offsetHeight;
    const midY = (box.minY + box.maxY) * scaleY / 2 - height / 2;
    const candidates = [
      { x: box.maxX * scaleX + 8, y: midY },
      { x: box.minX * scaleX - width - 8, y: midY },
      { x: box.minX * scaleX, y: box.minY * scaleY - height - 6 },
      { x: box.minX * scaleX, y: box.maxY * scaleY + 6 },
    ];
    // Prefer a nearby free label position; keep long names inside the image.
    for (let y = 6; y + height <= layer.clientHeight - 6; y += height + 5) candidates.push({ x: box.minX * scaleX, y });
    const clamp = (candidate) => ({
      x: Math.max(6, Math.min(layer.clientWidth - width - 6, candidate.x)),
      y: Math.max(6, Math.min(layer.clientHeight - height - 6, candidate.y)),
    });
    const inside = candidates.filter(p => p.x >= 6 && p.x + width <= layer.clientWidth - 6 && p.y >= 6 && p.y + height <= layer.clientHeight - 6);
    const options = [...inside, ...candidates.map(clamp)];
    const position = options.find(p => placed.every(r => p.x + width + 4 <= r.x || r.x + r.width + 4 <= p.x || p.y + height + 4 <= r.y || r.y + r.height + 4 <= p.y)) || clamp(candidates[0]);
    label.style.left = `${position.x}px`;
    label.style.top = `${position.y}px`;
    placed.push({ ...position, width, height });
  }
}

function drawInstances() {
  const list = $("instanceList");
  list.innerHTML = "";
  const frame = manifest.frames[currentFrameIndex];
  frame.instances.forEach((instance, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "instanceItem";
    button.dataset.instanceIndex = String(index);
    button.setAttribute("aria-controls", "imageCanvas maskLabels cloudCanvas");
    button.innerHTML = `
      <span class="selectionMark" aria-hidden="true">✓</span>
      <span class="swatch" style="background: rgb(${instance.color.join(",")})"></span>
      <span class="name">${escapeHtml(instance.instance_id)} · ${escapeHtml(instance.class_name)}</span>
      <span class="detail">${Number(instance.confidence).toFixed(3)} / ${Number(instance.point_count).toLocaleString()} pts</span>
    `;
    button.addEventListener("click", () => {
      if (selectedInstanceIndices.has(index)) selectedInstanceIndices.delete(index);
      else selectedInstanceIndices.add(index);
      setImageMode("selected");
    });
    list.appendChild(button);
  });
  updateInstanceSelection();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function cssEscape(value) {
  if (window.CSS && CSS.escape) return CSS.escape(value);
  return String(value).replace(/["\\]/g, "\\$&");
}

function initWebgl() {
  const canvas = $("cloudCanvas");
  const gl = canvas.getContext("webgl", { antialias: false });
  if (!gl) throw new Error("浏览器不支持 WebGL");
  const vertexSource = `
    attribute vec3 a_position;
    attribute vec3 a_color;
    uniform mat4 u_matrix;
    uniform float u_pointSize;
    varying vec3 v_color;
    void main() {
      gl_Position = u_matrix * vec4(a_position, 1.0);
      gl_PointSize = u_pointSize;
      v_color = a_color / 255.0;
    }
  `;
  const fragmentSource = `
    precision mediump float;
    varying vec3 v_color;
    void main() {
      vec2 c = gl_PointCoord - vec2(0.5);
      if (dot(c, c) > 0.25) discard;
      gl_FragColor = vec4(v_color, 0.95);
    }
  `;
  const program = createProgram(gl, vertexSource, fragmentSource);
  glState = {
    gl,
    canvas,
    program,
    positionLocation: gl.getAttribLocation(program, "a_position"),
    colorLocation: gl.getAttribLocation(program, "a_color"),
    matrixLocation: gl.getUniformLocation(program, "u_matrix"),
    pointSizeLocation: gl.getUniformLocation(program, "u_pointSize"),
    positionBuffer: gl.createBuffer(),
    colorBuffer: gl.createBuffer(),
    bgPositionBuffer: gl.createBuffer(),
    bgColorBuffer: gl.createBuffer(),
    drawCount: 0,
    bgCount: 0,
    rotationX: -0.35,
    rotationY: 0.45,
    distance: 3.2,
    panX: 0,
    panY: 0,
    center: [0, 0, 0],
    radius: 4,
  };
  setupMouse(canvas);
}

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const vertex = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
}

function setupMouse(canvas) {
  let drag = null;
  let lastX = 0;
  let lastY = 0;
  function endDrag(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const pointerId = drag.pointerId;
    drag = null;
    if (canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
  }
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("pointerdown", (event) => {
    if (drag || (event.button !== 0 && event.button !== 2)) return;
    event.preventDefault();
    drag = { pointerId: event.pointerId, button: event.button };
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if ((event.buttons & (drag.button === 2 ? 2 : 1)) === 0) {
      endDrag(event);
      return;
    }
    event.preventDefault();
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    if (drag.button === 2) {
      // Translate in the camera plane, scaled to the perspective view at its target.
      const unitsPerPixel = 2 * glState.distance * Math.tan(Math.PI / 8) / Math.max(canvas.clientHeight, 1);
      glState.panX += dx * unitsPerPixel;
      glState.panY -= dy * unitsPerPixel;
    } else {
      glState.rotationY += dx * 0.008;
      glState.rotationX += dy * 0.008;
    }
    renderCloud();
  });
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", endDrag);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    glState.distance *= Math.exp(event.deltaY * 0.001);
    glState.distance = Math.max(0.6, Math.min(60, glState.distance));
    renderCloud();
  }, { passive: false });
}

function updateCloudBuffers() {
  if (!currentCloud || !glState) return;
  const frame = manifest.frames[currentFrameIndex];
  const positions = [];
  const colors = [];
  const useInstanceColor = $("colorMode").checked;
  const selectionBits = selectedMaskBits();
  for (let i = 0; i < currentCloud.maskBits.length; i += 1) {
    const bits = imageMode === "selected" ? (currentCloud.maskBits[i] & selectionBits) >>> 0 : currentCloud.maskBits[i];
    if (bits === 0) continue;
    positions.push(
      currentCloud.positions[i * 3],
      currentCloud.positions[i * 3 + 1],
      currentCloud.positions[i * 3 + 2],
    );
    if (!useInstanceColor) {
      colors.push(currentCloud.colors[i * 3], currentCloud.colors[i * 3 + 1], currentCloud.colors[i * 3 + 2]);
    } else if (popcount32(bits) >= 2) {
      colors.push(175, 82, 222);
    } else {
      const instanceIndex = Math.floor(Math.log2(bits));
      const color = frame.instances[instanceIndex]?.color || [255, 255, 255];
      colors.push(...color);
    }
  }
  const gl = glState.gl;
  gl.bindBuffer(gl.ARRAY_BUFFER, glState.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, glState.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.STATIC_DRAW);
  glState.drawCount = positions.length / 3;

  if (backgroundCloud && glState.bgCount === 0) {
    const gray = new Uint8Array(backgroundCloud.count * 3);
    for (let i = 0; i < backgroundCloud.count; i += 1) {
      const base = i * 3;
      const value = Math.max(58, Math.min(150, Math.round((backgroundCloud.colors[base] + backgroundCloud.colors[base + 1] + backgroundCloud.colors[base + 2]) / 5)));
      gray[base] = value;
      gray[base + 1] = value;
      gray[base + 2] = value;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, glState.bgPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, backgroundCloud.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, glState.bgColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, gray, gl.STATIC_DRAW);
    glState.bgCount = backgroundCloud.count;
  }
  const min = manifest.bounds.minimum;
  const max = manifest.bounds.maximum;
  glState.center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  glState.radius = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1);
}

function resizeCanvasToDisplaySize(canvas) {
  const width = Math.max(2, Math.floor(canvas.clientWidth * window.devicePixelRatio));
  const height = Math.max(2, Math.floor(canvas.clientHeight * window.devicePixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function renderCloud() {
  if (!glState) return;
  const gl = glState.gl;
  resizeCanvasToDisplaySize(glState.canvas);
  gl.viewport(0, 0, glState.canvas.width, glState.canvas.height);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.02, 0.03, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(glState.program);
  const aspect = glState.canvas.width / glState.canvas.height;
  const matrix = computeMatrix(aspect);
  gl.uniformMatrix4fv(glState.matrixLocation, false, matrix);
  if ($("showBackground").checked && glState.bgCount) {
    drawBuffers(glState.bgPositionBuffer, glState.bgColorBuffer, glState.bgCount, 1.0);
  }
  if (glState.drawCount) {
    drawBuffers(glState.positionBuffer, glState.colorBuffer, glState.drawCount, 3.0);
  }
}

function drawBuffers(positionBuffer, colorBuffer, count, pointSize) {
  const gl = glState.gl;
  gl.uniform1f(glState.pointSizeLocation, pointSize * window.devicePixelRatio);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(glState.positionLocation);
  gl.vertexAttribPointer(glState.positionLocation, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.enableVertexAttribArray(glState.colorLocation);
  gl.vertexAttribPointer(glState.colorLocation, 3, gl.UNSIGNED_BYTE, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, count);
}

function computeMatrix(aspect) {
  const projection = perspective(Math.PI / 4, aspect, 0.01, 200);
  let view = identity();
  view = multiply(view, translate(glState.panX, glState.panY, -glState.distance));
  view = multiply(view, rotateX(glState.rotationX));
  view = multiply(view, rotateZ(glState.rotationY));
  view = multiply(view, scaleMatrix(2 / glState.radius, 2 / glState.radius, 2 / glState.radius));
  view = multiply(view, translate(-glState.center[0], -glState.center[1], -glState.center[2]));
  return multiply(projection, view);
}

function identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, (2 * far * near) * nf, 0]);
}

function translate(x, y, z) {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

function scaleMatrix(x, y, z) {
  return new Float32Array([x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1]);
}

function rotateX(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
}

function rotateY(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
}

function rotateZ(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function showError(error) {
  console.error(error);
  setStatus(error.message || String(error), true);
}

async function main() {
  try {
    manifest = window.MASK_PROJECTION_MANIFEST;
    if (!manifest) throw new Error("manifest.js 未定义 window.MASK_PROJECTION_MANIFEST");
    const select = $("frameSelect");
    manifest.frames.forEach((frame, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${index.toString().padStart(3, "0")} / ${frame.source_frame_id}`;
      select.appendChild(option);
    });
    $("frameSlider").max = String(manifest.frames.length - 1);
    setupControls();
    initWebgl();
    window.addEventListener("resize", () => {
      renderCloud();
      drawMaskLabels(currentMaskBounds);
    });
    await loadFrame(0);
  } catch (error) {
    showError(error);
  }
}

main();
