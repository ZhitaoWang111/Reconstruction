/* Single-object generation results: key frame, generated mesh, scan instance cloud.
 *
 * Three independent WebGL viewers share one canvas each. All are Z-up and draw
 * their own X/Y/Z axes at the local origin, which all data sets put under the
 * object's bottom centre:
 *   - overlay: scaled mesh (normalised × scale_factor → metres) + scan cloud together;
 *   - mesh:    SAM3D output turned Z-up, normalised, no unit;
 *   - cloud:   the handoff's instance points in the scan frame, metres, only shifted.
 * The page has to open over file:// with no network, so there is no library.
 */
(function () {
  "use strict";

  const DATA = window.PROBE || { objects: [] };
  const objects = DATA.objects.slice().sort(byObjectNumber);
  const $ = (id) => document.getElementById(id);

  function byObjectNumber(a, b) { return number(a.object_id) - number(b.object_id); }
  function number(objectId) {
    const digits = String(objectId).replace(/\D+/g, "");
    return digits === "" ? Number.MAX_SAFE_INTEGER : Number(digits);
  }
  function fmt(value, digits) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
  }
  function fmtInt(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN") : "—";
  }
  function triple(values, digits) {
    return Array.isArray(values) ? values.map((value) => fmt(value, digits)).join(" × ") : "—";
  }

  /* -------------------------------------------------------------- decoding */

  function decodeBase64(payload) {
    const binary = atob(payload);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }
  function float32(payload) { return new Float32Array(decodeBase64(payload).buffer); }

  const decoded = new Map();   // object_id -> { mesh, cloud }, decoded once on first view

  function decodeObject(object) {
    if (decoded.has(object.object_id)) return decoded.get(object.object_id);
    const result = { mesh: null, cloud: null };
    if (object.mesh) {
      const m = object.mesh;
      const indexBytes = decodeBase64(m.indices);
      result.mesh = {
        positions: float32(m.positions),
        normals: m.normals ? float32(m.normals) : null,
        colours: decodeBase64(m.colours),
        indices: m.index_bits === 32 ? new Uint32Array(indexBytes.buffer) : new Uint16Array(indexBytes.buffer),
        count: m.vertex_count,
        extents: m.extents,
      };
    }
    if (object.cloud) {
      const c = object.cloud;
      result.cloud = {
        positions: float32(c.positions),
        colours: decodeBase64(c.colours),
        count: c.point_count,
        extents: c.extents,
      };
    }
    decoded.set(object.object_id, result);
    return result;
  }

  /* ----------------------------------------------------------------- maths */
  // Column-major 4x4 matrices, as WebGL expects.

  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col += 1) {
      for (let row = 0; row < 4; row += 1) {
        let sum = 0;
        for (let k = 0; k < 4; k += 1) sum += a[k * 4 + row] * b[col * 4 + k];
        out[col * 4 + row] = sum;
      }
    }
    return out;
  }
  function perspective(fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }
  const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  function normalise(v) {
    const n = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
  }
  // World Z stays screen-up: the camera always orbits around the Z axis.
  function lookAt(eye, target, up) {
    const f = normalise(subtract(target, eye));
    const s = normalise(cross(f, up));
    const u = cross(s, f);
    return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
      -dot(s, eye), -dot(u, eye), dot(f, eye), 1]);
  }

  /* ---------------------------------------------------------------- viewer */

  const FOV = Math.PI / 4;
  const UP = [0, 0, 1];
  // Red / green / blue for X / Y / Z, the usual convention; each axis is also labelled.
  const AXIS_COLOURS = [[239, 84, 79], [104, 204, 92], [84, 140, 250]];

  const VERTEX_SHADER = `
    attribute vec3 a_position;
    attribute vec3 a_normal;
    attribute vec3 a_color;
    uniform mat4 u_matrix;
    uniform float u_lit;        // 1: shade with the normal, 0: use the colour as is
    uniform float u_flat;       // 1: replace the colour with u_flat_color
    uniform vec3 u_flat_color;
    uniform float u_point_size;
    varying vec3 v_color;
    void main() {
      vec3 base = mix(a_color, u_flat_color, u_flat);
      // Two-sided Lambert: SAM3D meshes are not guaranteed to have outward normals.
      float light = 0.32 + 0.68 * abs(dot(normalize(a_normal), normalize(vec3(0.45, -0.55, 0.75))));
      v_color = base * mix(1.0, light, u_lit);
      gl_PointSize = u_point_size;
      gl_Position = u_matrix * vec4(a_position, 1.0);
    }`;
  const FRAGMENT_SHADER = `
    precision mediump float;
    varying vec3 v_color;
    void main() { gl_FragColor = vec4(v_color, 1.0); }`;

  function linkProgram(gl, vertexSource, fragmentSource) {
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }

  // options: gridMinor / gridMajor spacing, axisLength(reach) -> length of each axis.
  function createViewer(canvas, labelLayer, options) {
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) return null;
    const uintIndices = gl.getExtension("OES_element_index_uint");
    const program = linkProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    const loc = {
      position: gl.getAttribLocation(program, "a_position"),
      normal: gl.getAttribLocation(program, "a_normal"),
      color: gl.getAttribLocation(program, "a_color"),
      matrix: gl.getUniformLocation(program, "u_matrix"),
      lit: gl.getUniformLocation(program, "u_lit"),
      flat: gl.getUniformLocation(program, "u_flat"),
      flatColor: gl.getUniformLocation(program, "u_flat_color"),
      pointSize: gl.getUniformLocation(program, "u_point_size"),
    };
    const state = {
      kind: null, buffers: {}, count: 0, indexCount: 0, indexType: gl.UNSIGNED_SHORT, colourStride: 3,
      grid: null, axes: null, axisLength: 1, extents: [1, 1, 1],
      camera: { yaw: 0, pitch: 0, distance: 3, target: [0, 0, 0] }, home: null,
      flat: false, ground: true, pointSize: 3, pending: false,
    };
    const axisLabels = ["X", "Y", "Z"].map((name) => {
      const span = document.createElement("span");
      span.className = `axis-label axis-${name.toLowerCase()}`;
      span.textContent = name;
      span.hidden = true;
      labelLayer.append(span);
      return span;
    });

    function buffer(data, target = gl.ARRAY_BUFFER) {
      const handle = gl.createBuffer();
      gl.bindBuffer(target, handle);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return handle;
    }

    // segments: [[from xyz], [to xyz], [r, g, b] 0-255]
    function lineSet(segments) {
      const positions = new Float32Array(segments.length * 6);
      const colours = new Uint8Array(segments.length * 6);
      segments.forEach(([from, to, colour], i) => {
        positions.set(from, i * 6); positions.set(to, i * 6 + 3);
        colours.set(colour, i * 6); colours.set(colour, i * 6 + 3);
      });
      return { position: buffer(positions), colour: buffer(colours), count: segments.length * 2 };
    }

    // Ground grid on Z=0; every gridMajor-th line is brighter.
    function buildGrid(half) {
      const segments = [];
      const steps = Math.round(half / options.gridMinor);
      const ratio = Math.max(1, Math.round(options.gridMajor / options.gridMinor));
      for (let i = -steps; i <= steps; i += 1) {
        const v = i * options.gridMinor;
        const colour = i % ratio === 0 ? [84, 95, 108] : [40, 47, 56];
        segments.push([[v, -half, 0], [v, half, 0], colour], [[-half, v, 0], [half, v, 0], colour]);
      }
      return lineSet(segments);
    }

    function buildAxes(length) {
      const lines = lineSet(AXIS_COLOURS.map((colour, axis) => {
        const tip = [0, 0, 0];
        tip[axis] = length;
        return [[0, 0, 0], tip, colour];
      }));
      // GL lines are one pixel wide everywhere; a run of points along each axis gives
      // them enough weight to read against the object.
      const per = 240;
      const positions = new Float32Array(3 * per * 3);
      const colours = new Uint8Array(3 * per * 3);
      for (let axis = 0; axis < 3; axis += 1) {
        for (let i = 0; i < per; i += 1) {
          const offset = (axis * per + i) * 3;
          positions[offset + axis] = (length * i) / (per - 1);
          colours.set(AXIS_COLOURS[axis], offset);
        }
      }
      return { lines, dots: { position: buffer(positions), colour: buffer(colours), count: 3 * per } };
    }
    function deleteSet(set) {
      if (!set) return;
      for (const key of ["position", "colour"]) if (set[key]) gl.deleteBuffer(set[key]);
    }
    function releaseContent() {
      for (const handle of Object.values(state.buffers)) gl.deleteBuffer(handle);
      state.buffers = {};
      state.kind = null;
    }

    // Size the axes, grid and home camera to the object, then reset the view.
    function frame(extents) {
      state.extents = extents.map((value) => Math.max(value, 1e-3));
      const longest = Math.max(...state.extents);
      state.axisLength = longest * 0.6;
      if (state.axes) { deleteSet(state.axes.lines); deleteSet(state.axes.dots); }
      state.axes = buildAxes(state.axisLength);
      const half = Math.max(options.gridMajor, Math.ceil((longest * 0.75) / options.gridMajor) * options.gridMajor);
      deleteSet(state.grid);
      state.grid = buildGrid(half);
      // Front-right, slightly above: X to the right, Y into the screen, Z up.
      state.home = { yaw: -1.1, pitch: 0.38, distance: longest * 2.4, target: [0, 0, state.extents[2] / 2] };
      reset();
    }

    function setMesh(mesh) {
      releaseContent();
      if (!mesh) { requestDraw(); return; }
      state.kind = "mesh";
      state.buffers.position = buffer(mesh.positions);
      if (mesh.normals) state.buffers.normal = buffer(mesh.normals);
      state.buffers.colour = buffer(mesh.colours);
      state.colourStride = Math.round(mesh.colours.length / mesh.count) || 3;
      state.buffers.index = buffer(mesh.indices, gl.ELEMENT_ARRAY_BUFFER);
      state.indexCount = mesh.indices.length;
      state.indexType = mesh.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      state.count = mesh.count;
      if (state.indexType === gl.UNSIGNED_INT && !uintIndices) state.kind = null;
      frame(mesh.extents);
    }

    function setCloud(cloud) {
      releaseContent();
      if (!cloud) { requestDraw(); return; }
      state.kind = "cloud";
      state.buffers.position = buffer(cloud.positions);
      state.buffers.colour = buffer(cloud.colours);
      state.colourStride = Math.round(cloud.colours.length / cloud.count) || 3;
      state.count = cloud.count;
      frame(cloud.extents);
    }

    function reset() {
      const home = state.home || { yaw: -1.1, pitch: 0.38, distance: 3, target: [0, 0, 0.5] };
      state.camera = { yaw: home.yaw, pitch: home.pitch, distance: home.distance, target: home.target.slice() };
      requestDraw();
    }

    function eye() {
      const { yaw, pitch, distance, target } = state.camera;
      return [
        target[0] + distance * Math.cos(pitch) * Math.cos(yaw),
        target[1] + distance * Math.cos(pitch) * Math.sin(yaw),
        target[2] + distance * Math.sin(pitch),
      ];
    }

    function bind(handle, location, size, type, normalised, stride) {
      if (location < 0) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, handle);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, type, normalised, stride || 0, 0);
    }
    function constant(location, values) {
      if (location < 0) return;
      gl.disableVertexAttribArray(location);
      gl.vertexAttrib3f(location, values[0], values[1], values[2]);
    }

    function drawLines(set, mode) {
      if (!set) return;
      bind(set.position, loc.position, 3, gl.FLOAT, false);
      bind(set.colour, loc.color, 3, gl.UNSIGNED_BYTE, true);
      constant(loc.normal, [0, 0, 1]);
      gl.uniform1f(loc.lit, 0);
      gl.uniform1f(loc.flat, 0);
      gl.drawArrays(mode, 0, set.count);
    }

    function drawContent(dpr) {
      if (!state.kind) return;
      bind(state.buffers.position, loc.position, 3, gl.FLOAT, false);
      bind(state.buffers.colour, loc.color, 3, gl.UNSIGNED_BYTE, true, state.colourStride);
      gl.uniform1f(loc.flat, state.flat ? 1 : 0);
      gl.uniform3f(loc.flatColor, 0.74, 0.79, 0.86);
      if (state.kind === "mesh") {
        if (state.buffers.normal) bind(state.buffers.normal, loc.normal, 3, gl.FLOAT, false);
        else constant(loc.normal, [0, 0, 1]);
        gl.uniform1f(loc.lit, state.buffers.normal ? 1 : 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state.buffers.index);
        gl.drawElements(gl.TRIANGLES, state.indexCount, state.indexType, 0);
      } else {
        constant(loc.normal, [0, 0, 1]);
        gl.uniform1f(loc.lit, 0);
        gl.uniform1f(loc.pointSize, state.pointSize * dpr);
        gl.drawArrays(gl.POINTS, 0, state.count);
      }
    }

    // Put the X / Y / Z text next to each axis tip, in CSS pixels over the canvas.
    function placeLabels(mvp) {
      const width = canvas.clientWidth, height = canvas.clientHeight;
      axisLabels.forEach((span, axis) => {
        const tip = [0, 0, 0];
        tip[axis] = state.axisLength * 1.08;
        const x = mvp[0] * tip[0] + mvp[4] * tip[1] + mvp[8] * tip[2] + mvp[12];
        const y = mvp[1] * tip[0] + mvp[5] * tip[1] + mvp[9] * tip[2] + mvp[13];
        const w = mvp[3] * tip[0] + mvp[7] * tip[1] + mvp[11] * tip[2] + mvp[15];
        if (w <= 1e-6) { span.hidden = true; return; }
        span.hidden = false;
        span.style.transform = `translate(${((x / w) * 0.5 + 0.5) * width}px, ${(0.5 - (y / w) * 0.5) * height}px)`;
      });
    }

    function draw() {
      state.pending = false;
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(2, Math.floor(canvas.clientWidth * dpr));
      const height = Math.max(2, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.078, 0.094, 0.114, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program);
      const longest = Math.max(...state.extents);
      const projection = perspective(FOV, width / height, longest * 0.01, longest * 200);
      const mvp = multiply(projection, lookAt(eye(), state.camera.target, UP));
      gl.uniformMatrix4fv(loc.matrix, false, mvp);
      gl.uniform1f(loc.pointSize, 1);

      gl.enable(gl.DEPTH_TEST);
      if (state.ground) drawLines(state.grid, gl.LINES);
      drawContent(dpr);
      // The origin sits under the object, so the axes are drawn on top of it.
      gl.disable(gl.DEPTH_TEST);
      if (state.axes) {
        drawLines(state.axes.lines, gl.LINES);
        gl.uniform1f(loc.pointSize, 2.5 * dpr);
        drawLines(state.axes.dots, gl.POINTS);
      }
      placeLabels(mvp);
    }

    function requestDraw() {
      if (state.pending) return;
      state.pending = true;
      window.requestAnimationFrame(draw);
    }

    // Left drag orbits, right drag pans, the wheel zooms.
    let drag = null;
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("mousedown", (event) => {
      drag = { x: event.clientX, y: event.clientY, pan: event.button === 2 };
      canvas.classList.add("dragging");
    });
    window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("dragging"); });
    window.addEventListener("mousemove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      drag.x = event.clientX; drag.y = event.clientY;
      const camera = state.camera;
      if (drag.pan) {
        const forward = normalise(subtract(camera.target, eye()));
        const right = normalise(cross(forward, UP));
        const up = cross(right, forward);
        const scale = (camera.distance * 1.1) / Math.max(1, canvas.clientHeight);
        for (let i = 0; i < 3; i += 1) camera.target[i] += (-dx * right[i] + dy * up[i]) * scale;
      } else {
        camera.yaw -= dx * 0.008;
        camera.pitch = Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.008));
      }
      requestDraw();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const longest = Math.max(...state.extents);
      state.camera.distance = Math.max(longest * 0.3, Math.min(longest * 20,
        state.camera.distance * Math.exp(event.deltaY * 0.0012)));
      requestDraw();
    }, { passive: false });
    window.addEventListener("resize", requestDraw);

    frame(state.extents);
    return {
      setMesh, setCloud, reset, requestDraw,
      setGround(value) { state.ground = value; requestDraw(); },
      setFlat(value) { state.flat = value; requestDraw(); },
      setPointSize(value) { state.pointSize = value; requestDraw(); },
    };
  }

  /* ------------------------------------------------------- overlay viewer */
  // One canvas showing the scaled mesh and the scan cloud together, so the
  // viewer can eyeball whether the auto scale factor actually fits the cloud.
  // The mesh is uploaded already multiplied by scale_factor (metres, Z-up),
  // matching the cloud's frame; the camera frames the union of both extents.

  function createOverlayViewer(canvas, labelLayer, options) {
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) return null;
    const uintIndices = gl.getExtension("OES_element_index_uint");
    const program = linkProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    const loc = {
      position: gl.getAttribLocation(program, "a_position"),
      normal: gl.getAttribLocation(program, "a_normal"),
      color: gl.getAttribLocation(program, "a_color"),
      matrix: gl.getUniformLocation(program, "u_matrix"),
      lit: gl.getUniformLocation(program, "u_lit"),
      flat: gl.getUniformLocation(program, "u_flat"),
      flatColor: gl.getUniformLocation(program, "u_flat_color"),
      pointSize: gl.getUniformLocation(program, "u_point_size"),
    };
    const state = {
      mesh: null, cloud: null, grid: null, axes: null, axisLength: 1,
      extents: [1, 1, 1], colourStride: 3,
      camera: { yaw: 0, pitch: 0, distance: 3, target: [0, 0, 0] }, home: null,
      flat: false, ground: true, showMesh: true, showCloud: true, pointSize: 3, pending: false,
    };
    const axisLabels = ["X", "Y", "Z"].map((name) => {
      const span = document.createElement("span");
      span.className = `axis-label axis-${name.toLowerCase()}`;
      span.textContent = name;
      span.hidden = true;
      labelLayer.append(span);
      return span;
    });

    function buffer(data, target = gl.ARRAY_BUFFER) {
      const handle = gl.createBuffer();
      gl.bindBuffer(target, handle);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return handle;
    }
    function lineSet(segments) {
      const positions = new Float32Array(segments.length * 6);
      const colours = new Uint8Array(segments.length * 6);
      segments.forEach(([from, to, colour], i) => {
        positions.set(from, i * 6); positions.set(to, i * 6 + 3);
        colours.set(colour, i * 6); colours.set(colour, i * 6 + 3);
      });
      return { position: buffer(positions), colour: buffer(colours), count: segments.length * 2 };
    }
    function buildGrid(half) {
      const segments = [];
      const steps = Math.round(half / options.gridMinor);
      const ratio = Math.max(1, Math.round(options.gridMajor / options.gridMinor));
      for (let i = -steps; i <= steps; i += 1) {
        const v = i * options.gridMinor;
        const colour = i % ratio === 0 ? [84, 95, 108] : [40, 47, 56];
        segments.push([[v, -half, 0], [v, half, 0], colour], [[-half, v, 0], [half, v, 0], colour]);
      }
      return lineSet(segments);
    }
    function buildAxes(length) {
      const lines = lineSet(AXIS_COLOURS.map((colour, axis) => {
        const tip = [0, 0, 0]; tip[axis] = length;
        return [[0, 0, 0], tip, colour];
      }));
      const per = 240;
      const positions = new Float32Array(3 * per * 3);
      const colours = new Uint8Array(3 * per * 3);
      for (let axis = 0; axis < 3; axis += 1) {
        for (let i = 0; i < per; i += 1) {
          const offset = (axis * per + i) * 3;
          positions[offset + axis] = (length * i) / (per - 1);
          colours.set(AXIS_COLOURS[axis], offset);
        }
      }
      return { lines, dots: { position: buffer(positions), colour: buffer(colours), count: 3 * per } };
    }
    function deleteSet(set) {
      if (!set) return;
      for (const key of ["position", "colour"]) if (set[key]) gl.deleteBuffer(set[key]);
    }
    function releaseLayer(layer) {
      if (!layer) return;
      for (const key of ["position", "normal", "colour", "index"]) if (layer[key]) gl.deleteBuffer(layer[key]);
    }

    function uploadMesh(mesh, scale) {
      releaseLayer(state.mesh);
      if (!mesh) { state.mesh = null; return; }
      // Multiply normalised positions by scale_factor on upload, in float32.
      const positions = new Float32Array(mesh.positions.length);
      for (let i = 0; i < mesh.positions.length; i += 1) positions[i] = mesh.positions[i] * scale;
      const indexType = mesh.indices instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      state.mesh = {
        position: buffer(positions),
        normal: mesh.normals ? buffer(mesh.normals) : null,
        colour: buffer(mesh.colours),
        index: buffer(mesh.indices, gl.ELEMENT_ARRAY_BUFFER),
        indexCount: mesh.indices.length,
        indexType,
        count: mesh.count,
        hasNormals: !!mesh.normals,
        colourStride: Math.round(mesh.colours.length / mesh.count) || 3,
      };
      if (indexType === gl.UNSIGNED_INT && !uintIndices) state.mesh = null;
    }
    function uploadCloud(cloud) {
      releaseLayer(state.cloud);
      if (!cloud) { state.cloud = null; return; }
      state.cloud = {
        position: buffer(cloud.positions),
        colour: buffer(cloud.colours),
        count: cloud.count,
        colourStride: Math.round(cloud.colours.length / cloud.count) || 3,
      };
    }

    function recomputeExtents() {
      const ext = [0, 0, 0];
      if (state.mesh) {
        // Scaled extents = original extents × scale; extents are stored in the
        // entry as the normalised XYZ span, already Z-up.
        for (let i = 0; i < 3; i += 1) ext[i] = Math.max(ext[i], (state.meshExtents[i] || 0));
      }
      if (state.cloud) {
        for (let i = 0; i < 3; i += 1) ext[i] = Math.max(ext[i], (state.cloudExtents[i] || 0));
      }
      state.extents = ext.map((v) => Math.max(v, 1e-3));
      const longest = Math.max(...state.extents);
      state.axisLength = longest * 0.6;
      if (state.axes) { deleteSet(state.axes.lines); deleteSet(state.axes.dots); }
      state.axes = buildAxes(state.axisLength);
      const half = Math.max(options.gridMajor, Math.ceil((longest * 0.75) / options.gridMajor) * options.gridMajor);
      deleteSet(state.grid);
      state.grid = buildGrid(half);
      state.home = { yaw: -1.1, pitch: 0.38, distance: longest * 2.4, target: [0, 0, state.extents[2] / 2] };
      reset();
    }

    function setOverlay(mesh, scale, cloud) {
      state.meshExtents = mesh ? (mesh.extents || [0, 0, 0]).slice() : [0, 0, 0];
      state.cloudExtents = cloud ? (cloud.extents || [0, 0, 0]).slice() : [0, 0, 0];
      uploadMesh(mesh, scale || 0);
      uploadCloud(cloud);
      recomputeExtents();
      requestDraw();
    }

    function reset() {
      const home = state.home || { yaw: -1.1, pitch: 0.38, distance: 3, target: [0, 0, 0.5] };
      state.camera = { yaw: home.yaw, pitch: home.pitch, distance: home.distance, target: home.target.slice() };
      requestDraw();
    }
    function eye() {
      const { yaw, pitch, distance, target } = state.camera;
      return [
        target[0] + distance * Math.cos(pitch) * Math.cos(yaw),
        target[1] + distance * Math.cos(pitch) * Math.sin(yaw),
        target[2] + distance * Math.sin(pitch),
      ];
    }
    function bind(handle, location, size, type, normalised, stride) {
      if (location < 0) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, handle);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, type, normalised, stride || 0, 0);
    }
    function constant(location, values) {
      if (location < 0) return;
      gl.disableVertexAttribArray(location);
      gl.vertexAttrib3f(location, values[0], values[1], values[2]);
    }
    function drawLines(set, mode) {
      if (!set) return;
      bind(set.position, loc.position, 3, gl.FLOAT, false);
      bind(set.colour, loc.color, 3, gl.UNSIGNED_BYTE, true);
      constant(loc.normal, [0, 0, 1]);
      gl.uniform1f(loc.lit, 0); gl.uniform1f(loc.flat, 0);
      gl.drawArrays(mode, 0, set.count);
    }
    function drawMeshLayer(dpr) {
      const m = state.mesh; if (!m) return;
      bind(m.position, loc.position, 3, gl.FLOAT, false);
      bind(m.colour, loc.color, 3, gl.UNSIGNED_BYTE, true, m.colourStride);
      gl.uniform1f(loc.flat, state.flat ? 1 : 0);
      gl.uniform3f(loc.flatColor, 0.74, 0.79, 0.86);
      if (m.normal) bind(m.normal, loc.normal, 3, gl.FLOAT, false);
      else constant(loc.normal, [0, 0, 1]);
      gl.uniform1f(loc.lit, m.hasNormals ? 1 : 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.index);
      gl.drawElements(gl.TRIANGLES, m.indexCount, m.indexType, 0);
    }
    function drawCloudLayer(dpr) {
      const c = state.cloud; if (!c) return;
      bind(c.position, loc.position, 3, gl.FLOAT, false);
      bind(c.colour, loc.color, 3, gl.UNSIGNED_BYTE, true, c.colourStride);
      constant(loc.normal, [0, 0, 1]);
      gl.uniform1f(loc.lit, 0); gl.uniform1f(loc.flat, 0);
      gl.uniform1f(loc.pointSize, state.pointSize * dpr);
      gl.drawArrays(gl.POINTS, 0, c.count);
    }
    function placeLabels(mvp) {
      const width = canvas.clientWidth, height = canvas.clientHeight;
      axisLabels.forEach((span, axis) => {
        const tip = [0, 0, 0]; tip[axis] = state.axisLength * 1.08;
        const x = mvp[0] * tip[0] + mvp[4] * tip[1] + mvp[8] * tip[2] + mvp[12];
        const y = mvp[1] * tip[0] + mvp[5] * tip[1] + mvp[9] * tip[2] + mvp[13];
        const w = mvp[3] * tip[0] + mvp[7] * tip[1] + mvp[11] * tip[2] + mvp[15];
        if (w <= 1e-6) { span.hidden = true; return; }
        span.hidden = false;
        span.style.transform = `translate(${((x / w) * 0.5 + 0.5) * width}px, ${(0.5 - (y / w) * 0.5) * height}px)`;
      });
    }
    function draw() {
      state.pending = false;
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(2, Math.floor(canvas.clientWidth * dpr));
      const height = Math.max(2, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.078, 0.094, 0.114, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program);
      const longest = Math.max(...state.extents);
      const projection = perspective(FOV, width / height, longest * 0.01, longest * 200);
      const mvp = multiply(projection, lookAt(eye(), state.camera.target, UP));
      gl.uniformMatrix4fv(loc.matrix, false, mvp);
      gl.uniform1f(loc.pointSize, 1);
      gl.enable(gl.DEPTH_TEST);
      if (state.ground) drawLines(state.grid, gl.LINES);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      if (state.showCloud) drawCloudLayer(dpr);
      gl.disable(gl.BLEND);
      if (state.showMesh) drawMeshLayer(dpr);
      gl.disable(gl.DEPTH_TEST);
      if (state.axes) {
        drawLines(state.axes.lines, gl.LINES);
        gl.uniform1f(loc.pointSize, 2.5 * dpr);
        drawLines(state.axes.dots, gl.POINTS);
      }
      placeLabels(mvp);
    }
    function requestDraw() {
      if (state.pending) return;
      state.pending = true;
      window.requestAnimationFrame(draw);
    }
    let drag = null;
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("mousedown", (event) => {
      drag = { x: event.clientX, y: event.clientY, pan: event.button === 2 };
      canvas.classList.add("dragging");
    });
    window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("dragging"); });
    window.addEventListener("mousemove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      drag.x = event.clientX; drag.y = event.clientY;
      const camera = state.camera;
      if (drag.pan) {
        const forward = normalise(subtract(camera.target, eye()));
        const right = normalise(cross(forward, UP));
        const up = cross(right, forward);
        const scale = (camera.distance * 1.1) / Math.max(1, canvas.clientHeight);
        for (let i = 0; i < 3; i += 1) camera.target[i] += (-dx * right[i] + dy * up[i]) * scale;
      } else {
        camera.yaw -= dx * 0.008;
        camera.pitch = Math.max(-1.45, Math.min(1.45, camera.pitch + dy * 0.008));
      }
      requestDraw();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const longest = Math.max(...state.extents);
      state.camera.distance = Math.max(longest * 0.3, Math.min(longest * 20,
        state.camera.distance * Math.exp(event.deltaY * 0.0012)));
      requestDraw();
    }, { passive: false });
    window.addEventListener("resize", requestDraw);

    recomputeExtents();
    return {
      setOverlay, reset, requestDraw,
      setGround(value) { state.ground = value; requestDraw(); },
      setFlat(value) { state.flat = value; requestDraw(); },
      setPointSize(value) { state.pointSize = value; requestDraw(); },
      setShowMesh(value) { state.showMesh = value; requestDraw(); },
      setShowCloud(value) { state.showCloud = value; requestDraw(); },
    };
  }

  /* ------------------------------------------------------------------ page */

  const overlayViewer = createOverlayViewer($("overlayCanvas"), $("overlayAxisLabels"), { gridMinor: 0.1, gridMajor: 1 });
  const meshViewer = createViewer($("meshCanvas"), $("meshAxisLabels"), { gridMinor: 0.1, gridMajor: 0.5 });
  if (!meshViewer || !overlayViewer) $("fallback").hidden = false;

  function drawHeader() {
    $("objectCount").textContent = String(objects.length);
    const seconds = objects.reduce((sum, o) => sum + (o.report?.elapsed_seconds || 0), 0);
    $("totalSeconds").textContent = seconds ? `${fmt(seconds / 60, 1)} 分钟` : "—";
    const vram = Math.max(0, ...objects.map((o) => o.report?.peak_vram_gib || 0));
    $("peakVram").textContent = vram ? `${fmt(vram, 2)} GiB` : "—";
  }

  let selectedId = null;

  function visibleObjects() {
    const needle = $("search").value.trim().toLowerCase();
    if (!needle) return objects;
    return objects.filter((o) => o.object_id.toLowerCase().includes(needle)
      || String(o.label || "").toLowerCase().includes(needle));
  }

  function drawList() {
    const list = $("objectList");
    const shown = visibleObjects();
    list.replaceChildren();
    for (const object of shown) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "object-item";
      button.setAttribute("aria-current", object.object_id === selectedId ? "true" : "false");
      const label = document.createElement("span");
      label.className = "item-label";
      label.textContent = object.label || "（无标签）";
      const id = document.createElement("span");
      id.className = "item-id";
      id.textContent = object.object_id;
      button.append(label, id);
      if (object.report && object.report.watertight === false) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = "非水密";
        button.append(badge);
      }
      button.addEventListener("click", () => select(object.object_id));
      item.append(button);
      list.append(item);
    }
    $("filterNote").textContent = shown.length === objects.length ? `共 ${objects.length} 个` : `${shown.length} / ${objects.length} 个`;
  }

  function fact(name, value) {
    const row = document.createElement("div");
    row.className = "fact";
    const term = document.createElement("dt");
    term.textContent = name;
    const data = document.createElement("dd");
    data.textContent = value;
    row.append(term, data);
    return row;
  }

  function drawFacts(object) {
    const report = object.report || {};
    const bounds = object.handoff_bounds;
    const scanSize = bounds && bounds.minimum && bounds.maximum
      ? bounds.maximum.map((value, i) => value - bounds.minimum[i]) : null;
    const size = object.size || {};
    const scaledExtents = object.mesh?.extents
      ? object.mesh.extents.map((v) => v * (size.scale_factor || 0)) : null;
    const reviewText = size.needs_review === false ? "否"
      : size.needs_review === true ? `是（${(size.flags || []).join("；") || "—"}）` : "—";
    const groups = [
      ["生成", [
        ["网格尺寸（归一化）", triple(object.mesh?.extents, 3)],
        ["顶点 / 面", `${fmtInt(report.vertices)} / ${fmtInt(report.faces)}`],
        ["水密", report.watertight === true ? "是" : report.watertight === false ? "否" : "—"],
        ["耗时", report.elapsed_seconds != null ? `${fmt(report.elapsed_seconds, 1)} s` : "—"],
        ["显存峰值", report.peak_vram_gib != null ? `${fmt(report.peak_vram_gib, 2)} GiB` : "—"],
        ["seed", report.seed != null ? String(report.seed) : "—"],
      ]],
      ["缩放", [
        ["缩放系数", size.scale_factor != null ? fmt(size.scale_factor, 4) : "—"],
        ["缩放后高度", size.height_m != null ? `${fmt(size.height_m, 3)} m` : "—"],
        ["缩放后尺寸", scaledExtents ? `${triple(scaledExtents, 3)} m` : "—"],
        ["来源", size.source || "—"],
        ["需复核", reviewText],
        ["备注", size.override_note || "—"],
      ]],
      ["实例点云（扫描坐标系，米）", [
        ["点云尺寸", object.cloud ? `${triple(object.cloud.extents, 3)} m` : "—"],
        ["包围盒尺寸", scanSize ? `${triple(scanSize, 3)} m` : "—"],
        ["点数", object.cloud ? `${fmtInt(object.cloud.point_count)}${object.cloud.point_count_total > object.cloud.point_count ? `（原 ${fmtInt(object.cloud.point_count_total)}）` : ""}` : "—"],
        ["局部原点在扫描中的位置", object.cloud?.origin_scan_m ? `${triple(object.cloud.origin_scan_m, 3)} m` : "—"],
      ]],
      ["关键帧", [
        ["seq / 源帧", `${object.seq ?? "—"} / ${object.source_frame_id ?? "—"}`],
        ["mask 像素", fmtInt(object.visible_pixels)],
        ["结果目录", object.run_dir || "—"],
      ]],
    ];
    const container = $("facts");
    container.replaceChildren();
    for (const [title, rows] of groups) {
      const section = document.createElement("section");
      const heading = document.createElement("h3");
      heading.textContent = title;
      const list = document.createElement("dl");
      for (const [name, value] of rows) list.append(fact(name, value));
      section.append(heading, list);
      container.append(section);
    }
  }

  function select(objectId) {
    const object = objects.find((o) => o.object_id === objectId);
    if (!object) return;
    selectedId = objectId;
    const head = $("detailHead");
    head.replaceChildren();
    const title = document.createElement("h2");
    title.textContent = object.label || "（无标签）";
    const sub = document.createElement("span");
    sub.className = "sub";
    sub.textContent = object.object_id;
    head.append(title, sub);

    $("rgbImage").src = object.images?.rgb || "";
    $("maskImage").src = object.images?.mask || "";

    const data = decodeObject(object);
    if (meshViewer) meshViewer.setMesh(data.mesh);
    if (overlayViewer) {
      const scale = object.size?.scale_factor || 0;
      overlayViewer.setOverlay(data.mesh, scale, data.cloud);
    }
    drawFacts(object);
    drawList();

    const url = new URL(window.location.href);
    url.searchParams.set("object", objectId);
    window.history.replaceState(null, "", url);
  }

  $("search").addEventListener("input", drawList);
  $("meshGround").addEventListener("change", (e) => meshViewer && meshViewer.setGround(e.target.checked));
  $("meshFlat").addEventListener("change", (e) => meshViewer && meshViewer.setFlat(e.target.checked));
  $("overlayGround").addEventListener("change", (e) => overlayViewer && overlayViewer.setGround(e.target.checked));
  $("overlayFlat").addEventListener("change", (e) => overlayViewer && overlayViewer.setFlat(e.target.checked));
  $("overlayPointSize").addEventListener("input", (e) => overlayViewer && overlayViewer.setPointSize(Number(e.target.value)));
  $("overlayShowMesh").addEventListener("change", (e) => overlayViewer && overlayViewer.setShowMesh(e.target.checked));
  $("overlayShowCloud").addEventListener("change", (e) => overlayViewer && overlayViewer.setShowCloud(e.target.checked));
  for (const button of document.querySelectorAll("[data-reset]")) {
    button.addEventListener("click", () => {
      const viewer = button.dataset.reset === "mesh" ? meshViewer : overlayViewer;
      if (viewer) viewer.reset();
    });
  }

  drawHeader();
  const requested = new URLSearchParams(window.location.search).get("object");
  const initial = objects.find((o) => o.object_id === requested) || objects[0];
  if (initial) select(initial.object_id);
  else drawList();
})();

