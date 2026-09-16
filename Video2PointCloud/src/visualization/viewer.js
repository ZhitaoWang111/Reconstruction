(function () {
  "use strict";

  const payloads = window.POINT_CLOUD_PAYLOADS || {};
  const viewers = [];
  const state = {
    sync: true,
    pointSize: 1.35,
    colorMode: 0,
    autoRotate: false,
  };
  const comparedPayloads = [...document.querySelectorAll(".method-card")]
    .map((card) => payloads[card.dataset.method])
    .filter(Boolean);
  const sharedMinimum = [0, 1, 2].map((axis) =>
    Math.min(...comparedPayloads.map((payload) => payload.minimum[axis]))
  );
  const sharedMaximum = [0, 1, 2].map((axis) =>
    Math.max(...comparedPayloads.map((payload) => payload.maximum[axis]))
  );
  const sharedExtent = sharedMaximum.map((value, axis) => Math.max(value - sharedMinimum[axis], 1e-6));
  const sharedCenter = sharedMaximum.map((value, axis) => (value + sharedMinimum[axis]) * 0.5);
  const sharedRadius = Math.max(...sharedExtent) * 0.5;

  const vertexShaderSource = `
    attribute vec3 aPosition;
    attribute vec3 aColor;
    attribute float aHeight;
    uniform mat4 uProjection;
    uniform float uYaw;
    uniform float uPitch;
    uniform float uDistance;
    uniform vec2 uPan;
    uniform float uPointSize;
    uniform float uPixelRatio;
    uniform float uColorMode;
    varying vec3 vColor;
    vec3 heightColor(float t) {
      vec3 low = vec3(0.11, 0.47, 0.72);
      vec3 mid = vec3(0.25, 0.92, 0.64);
      vec3 high = vec3(0.98, 0.76, 0.28);
      return t < 0.5 ? mix(low, mid, t * 2.0) : mix(mid, high, (t - 0.5) * 2.0);
    }
    void main() {
      float cz = cos(uYaw), sz = sin(uYaw);
      float cx = cos(uPitch), sx = sin(uPitch);
      vec3 rz = vec3(cz * aPosition.x - sz * aPosition.y,
                     sz * aPosition.x + cz * aPosition.y,
                     aPosition.z);
      vec3 p = vec3(rz.x, cx * rz.y - sx * rz.z, sx * rz.y + cx * rz.z);
      p.xy += uPan;
      p.z -= uDistance;
      gl_Position = uProjection * vec4(p, 1.0);
      gl_PointSize = uPointSize * uPixelRatio * clamp(2.4 / max(-p.z, 0.2), 0.55, 3.5);
      vColor = mix(aColor, heightColor(aHeight), uColorMode);
    }
  `;

  const fragmentShaderSource = `
    precision mediump float;
    varying vec3 vColor;
    void main() {
      vec2 centered = gl_PointCoord - vec2(0.5);
      if (dot(centered, centered) > 0.25) discard;
      gl_FragColor = vec4(vColor, 0.94);
    }
  `;

  function decodeBase64(value) {
    const raw = atob(value);
    const bytes = new Uint8Array(raw.length);
    const block = 1 << 15;
    for (let start = 0; start < raw.length; start += block) {
      const end = Math.min(start + block, raw.length);
      for (let index = start; index < end; index += 1) bytes[index] = raw.charCodeAt(index);
    }
    return bytes;
  }

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || "WebGL shader compilation failed");
    }
    return shader;
  }

  function makeProgram(gl) {
    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource));
    gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || "WebGL program link failed");
    }
    return program;
  }

  function perspective(fov, aspect, near, far) {
    const f = 1 / Math.tan(fov / 2);
    const range = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (near + far) * range, -1,
      0, 0, 2 * near * far * range, 0,
    ]);
  }

  function decodedGeometry(payload) {
    const positionBytes = decodeBase64(payload.positions_u16_base64);
    const quantized = new Uint16Array(
      positionBytes.buffer,
      positionBytes.byteOffset,
      positionBytes.byteLength / 2
    );
    const colorBytes = decodeBase64(payload.colors_u8_base64);
    const positions = new Float32Array(payload.count * 3);
    const heights = new Float32Array(payload.count);
    const colors = new Float32Array(payload.count * 3);
    const minimum = payload.minimum;
    const extent = payload.maximum.map((value, index) => Math.max(value - minimum[index], 1e-6));
    for (let index = 0; index < payload.count; index += 1) {
      for (let axis = 0; axis < 3; axis += 1) {
        const offset = index * 3 + axis;
        const actual = minimum[axis] + (quantized[offset] / 65535) * extent[axis];
        positions[offset] = (actual - sharedCenter[axis]) / sharedRadius;
        colors[offset] = colorBytes[offset] / 255;
      }
      const actualHeight = minimum[2] + (quantized[index * 3 + 2] / 65535) * extent[2];
      heights[index] = (actualHeight - sharedMinimum[2]) / sharedExtent[2];
    }
    return { positions, colors, heights };
  }

  class PointCloudViewer {
    constructor(card, payload) {
      this.card = card;
      this.canvas = card.querySelector("canvas");
      this.payload = payload;
      this.camera = { yaw: -0.72, pitch: -0.88, distance: 3.15, panX: 0, panY: 0 };
      this.drag = null;
      this.gl = this.canvas.getContext("webgl", { antialias: false, alpha: true });
      if (!this.gl) throw new Error("浏览器没有可用的 WebGL 上下文");
      this.program = makeProgram(this.gl);
      this.locations = {
        position: this.gl.getAttribLocation(this.program, "aPosition"),
        color: this.gl.getAttribLocation(this.program, "aColor"),
        height: this.gl.getAttribLocation(this.program, "aHeight"),
        projection: this.gl.getUniformLocation(this.program, "uProjection"),
        yaw: this.gl.getUniformLocation(this.program, "uYaw"),
        pitch: this.gl.getUniformLocation(this.program, "uPitch"),
        distance: this.gl.getUniformLocation(this.program, "uDistance"),
        pan: this.gl.getUniformLocation(this.program, "uPan"),
        pointSize: this.gl.getUniformLocation(this.program, "uPointSize"),
        pixelRatio: this.gl.getUniformLocation(this.program, "uPixelRatio"),
        colorMode: this.gl.getUniformLocation(this.program, "uColorMode"),
      };
      this.installInteractions();
    }

    async initialize() {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const geometry = decodedGeometry(this.payload);
      this.upload(this.locations.position, geometry.positions, 3);
      this.upload(this.locations.color, geometry.colors, 3);
      this.upload(this.locations.height, geometry.heights, 1);
      this.card.querySelector(".point-count").textContent = `${this.payload.count.toLocaleString("zh-CN")} 点`;
      this.card.classList.add("ready");
      this.card.querySelector(".loading").classList.add("done");
      this.render();
    }

    upload(location, data, size) {
      const buffer = this.gl.createBuffer();
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
      this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
      this.gl.enableVertexAttribArray(location);
      this.gl.vertexAttribPointer(location, size, this.gl.FLOAT, false, 0, 0);
    }

    installInteractions() {
      this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
      this.canvas.addEventListener("pointerdown", (event) => {
        this.canvas.setPointerCapture(event.pointerId);
        this.canvas.classList.add("dragging");
        this.drag = { x: event.clientX, y: event.clientY, pan: event.button === 2 || event.shiftKey };
      });
      this.canvas.addEventListener("pointermove", (event) => {
        if (!this.drag) return;
        const dx = event.clientX - this.drag.x;
        const dy = event.clientY - this.drag.y;
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
        if (this.drag.pan) {
          this.camera.panX += dx * 0.0028 * this.camera.distance;
          this.camera.panY -= dy * 0.0028 * this.camera.distance;
        } else {
          this.camera.yaw += dx * 0.007;
          this.camera.pitch = Math.max(-1.55, Math.min(1.55, this.camera.pitch + dy * 0.007));
        }
        this.changed();
      });
      const endDrag = () => { this.drag = null; this.canvas.classList.remove("dragging"); };
      this.canvas.addEventListener("pointerup", endDrag);
      this.canvas.addEventListener("pointercancel", endDrag);
      this.canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        this.camera.distance = Math.max(1.15, Math.min(8, this.camera.distance * Math.exp(event.deltaY * 0.001)));
        this.changed();
      }, { passive: false });
      this.canvas.addEventListener("dblclick", () => { this.setPreset("perspective"); this.changed(); });
      this.card.querySelector(".fullscreen-button").addEventListener("click", () => {
        if (document.fullscreenElement === this.card) document.exitFullscreen();
        else this.card.requestFullscreen();
      });
    }

    changed() {
      if (state.sync) {
        for (const viewer of viewers) {
          if (viewer !== this) viewer.camera = { ...this.camera };
        }
      }
      renderAll();
    }

    setPreset(name) {
      const presets = {
        perspective: { yaw: -0.72, pitch: -0.88, distance: 3.15, panX: 0, panY: 0 },
        top: { yaw: 0, pitch: -1.545, distance: 3.0, panX: 0, panY: 0 },
        side: { yaw: -1.57, pitch: 0, distance: 3.05, panX: 0, panY: 0 },
      };
      this.camera = { ...presets[name] };
    }

    render() {
      if (!this.card.classList.contains("ready")) return;
      const gl = this.gl;
      const rect = this.canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(rect.width * pixelRatio));
      const height = Math.max(1, Math.round(rect.height * pixelRatio));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.02, 0.045, 0.04, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(this.program);
      gl.uniformMatrix4fv(this.locations.projection, false, perspective(Math.PI / 4, width / height, 0.02, 30));
      gl.uniform1f(this.locations.yaw, this.camera.yaw);
      gl.uniform1f(this.locations.pitch, this.camera.pitch);
      gl.uniform1f(this.locations.distance, this.camera.distance);
      gl.uniform2f(this.locations.pan, this.camera.panX, this.camera.panY);
      gl.uniform1f(this.locations.pointSize, state.pointSize);
      gl.uniform1f(this.locations.pixelRatio, pixelRatio);
      gl.uniform1f(this.locations.colorMode, state.colorMode);
      gl.drawArrays(gl.POINTS, 0, this.payload.count);
    }
  }

  function renderAll() { for (const viewer of viewers) viewer.render(); }

  function selectButtons(selector, attribute, value) {
    for (const button of document.querySelectorAll(selector)) {
      button.classList.toggle("active", button.dataset[attribute] === value);
    }
  }

  function runInteractionSelfTest() {
    const topButton = document.querySelector('[data-view="top"]');
    const heightButton = document.querySelector('[data-color="height"]');
    const rgbButton = document.querySelector('[data-color="rgb"]');
    const pointSize = document.querySelector("#point-size");
    topButton.click();
    if (!viewers.every((viewer) => Math.abs(viewer.camera.pitch + 1.545) < 1e-6)) {
      throw new Error("视角预设自检失败");
    }
    heightButton.click();
    if (state.colorMode !== 1) throw new Error("高度着色自检失败");
    pointSize.value = "2.1";
    pointSize.dispatchEvent(new Event("input"));
    if (state.pointSize !== 2.1) throw new Error("点大小自检失败");
    viewers[0].camera.yaw += 0.2;
    viewers[0].changed();
    if (!viewers.every((viewer) => Math.abs(viewer.camera.yaw - viewers[0].camera.yaw) < 1e-6)) {
      throw new Error("同步视角自检失败");
    }
    for (const viewer of viewers) {
      viewer.gl.finish();
      if (viewer.gl.getError() !== viewer.gl.NO_ERROR) throw new Error("WebGL 绘制自检失败");
      viewer.setPreset("perspective");
    }
    rgbButton.click();
    pointSize.value = "1.35";
    pointSize.dispatchEvent(new Event("input"));
    selectButtons("[data-view]", "view", "perspective");
    renderAll();
    document.documentElement.dataset.interactionSelftest = "pass";
  }

  async function boot() {
    try {
      for (const card of document.querySelectorAll(".method-card")) {
        const method = card.dataset.method;
        if (!payloads[method]) throw new Error(`缺少 ${method} 点云资源`);
        viewers.push(new PointCloudViewer(card, payloads[method]));
      }
      for (const viewer of viewers) await viewer.initialize();
      document.documentElement.dataset.viewerStatus = "ready";
      if (new URLSearchParams(window.location.search).has("selftest")) runInteractionSelfTest();
    } catch (error) {
      document.documentElement.dataset.viewerStatus = "error";
      for (const loading of document.querySelectorAll(".loading")) loading.textContent = error.message;
      console.error(error);
    }
  }

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      for (const viewer of viewers) viewer.setPreset(button.dataset.view);
      selectButtons("[data-view]", "view", button.dataset.view);
      renderAll();
    });
  });
  document.querySelectorAll("[data-color]").forEach((button) => {
    button.addEventListener("click", () => {
      state.colorMode = button.dataset.color === "height" ? 1 : 0;
      selectButtons("[data-color]", "color", button.dataset.color);
      renderAll();
    });
  });
  document.querySelector("#sync-view").addEventListener("change", (event) => { state.sync = event.target.checked; });
  document.querySelector("#point-size").addEventListener("input", (event) => { state.pointSize = Number(event.target.value); renderAll(); });
  document.querySelector("#auto-rotate").addEventListener("change", (event) => { state.autoRotate = event.target.checked; });
  document.querySelector("#reset-all").addEventListener("click", () => {
    for (const viewer of viewers) viewer.setPreset("perspective");
    selectButtons("[data-view]", "view", "perspective");
    renderAll();
  });
  window.addEventListener("resize", renderAll);
  document.addEventListener("fullscreenchange", () => setTimeout(renderAll, 60));

  let previous = performance.now();
  function animate(now) {
    const delta = Math.min((now - previous) / 1000, 0.05);
    previous = now;
    if (state.autoRotate && viewers.length) {
      viewers[0].camera.yaw += delta * 0.18;
      if (state.sync) {
        for (let index = 1; index < viewers.length; index += 1) viewers[index].camera = { ...viewers[0].camera };
      } else {
        for (let index = 1; index < viewers.length; index += 1) viewers[index].camera.yaw += delta * 0.18;
      }
      renderAll();
    }
    requestAnimationFrame(animate);
  }

  boot();
  requestAnimationFrame(animate);
})();
