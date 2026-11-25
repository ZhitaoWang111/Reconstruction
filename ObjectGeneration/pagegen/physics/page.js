(function () {
"use strict";
/* Physics & appearance review page – reuses the probe page viewer architecture */

const DATA = window.PHYSICS || { objects: [] };
const objects = DATA.objects.slice().sort((a, b) => {
  const n = s => parseInt(String(s).replace(/\D+/g,""), 10) || 0;
  return n(a.object_id) - n(b.object_id);
});
const $ = id => document.getElementById(id);

/* ── decoding ── */
function decodeBase64(payload) {
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
function float32(payload) { return new Float32Array(decodeBase64(payload).buffer); }

const decoded = new Map();
function decodeObject(obj) {
  if (decoded.has(obj.object_id)) return decoded.get(obj.object_id);
  const m = obj.mesh;
  const indexBytes = decodeBase64(m.indices);
  const result = {
    positions: float32(m.positions),
    normals: m.normals ? float32(m.normals) : null,
    colours: decodeBase64(m.colours),
    indices: m.index_bits === 32 ? new Uint32Array(indexBytes.buffer) : new Uint16Array(indexBytes.buffer),
    count: m.vertex_count,
    extents: m.extents,
  };
  decoded.set(obj.object_id, result);
  return result;
}

/* ── maths (column-major, same as probe page) ── */
function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++)
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k*4+row] * b[col*4+k];
      out[col*4+row] = sum;
    }
  return out;
}
function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
const subtract = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
function normalise(v) { const n = Math.hypot(v[0],v[1],v[2])||1; return [v[0]/n,v[1]/n,v[2]/n]; }
function lookAt(eye, target, up) {
  const f = normalise(subtract(target, eye));
  const s = normalise(cross(f, up));
  const u = cross(s, f);
  return new Float32Array([s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0,
    -dot(s,eye),-dot(u,eye),dot(f,eye),1]);
}

const FOV = Math.PI / 4;
const UP = [0, 0, 1];
const AXIS_COLOURS = [[239,84,79],[104,204,92],[84,140,250]];

const VERTEX_SHADER = `
  attribute vec3 a_position,a_normal,a_color;
  uniform mat4 u_matrix;uniform float u_lit,u_flat,u_point_size;uniform vec3 u_flat_color;
  varying vec3 v_color;
  void main(){
    vec3 base=mix(a_color,u_flat_color,u_flat);
    float light=0.32+0.68*abs(dot(normalize(a_normal),normalize(vec3(0.45,-0.55,0.75))));
    v_color=base*mix(1.0,light,u_lit);
    gl_PointSize=u_point_size;gl_Position=u_matrix*vec4(a_position,1.0);}`;
const FRAGMENT_SHADER = `precision mediump float;varying vec3 v_color;
  void main(){gl_FragColor=vec4(v_color,1.0);}`;

function linkProgram(gl, vs, fs) {
  const compile = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

function createViewer(canvas, labelLayer, options) {
  const gl = canvas.getContext("webgl", { antialias: true });
  if (!gl) return null;
  const uintIndices = gl.getExtension("OES_element_index_uint");
  const program = linkProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  const loc = {
    position: gl.getAttribLocation(program, "a_position"),
    normal:   gl.getAttribLocation(program, "a_normal"),
    color:    gl.getAttribLocation(program, "a_color"),
    matrix:   gl.getUniformLocation(program, "u_matrix"),
    lit:      gl.getUniformLocation(program, "u_lit"),
    flat:     gl.getUniformLocation(program, "u_flat"),
    flatColor:gl.getUniformLocation(program, "u_flat_color"),
    pointSize:gl.getUniformLocation(program, "u_point_size"),
  };
  const state = {
    kind: null, buffers: {}, count: 0, indexCount: 0, indexType: gl.UNSIGNED_SHORT,
    colourStride: 3, grid: null, axes: null, axisLength: 1, extents: [1,1,1],
    camera: { yaw: 0, pitch: 0, distance: 3, target: [0,0,0] }, home: null,
    flat: false, ground: true, pointSize: 3, pending: false,
  };
  const axisLabels = ["X","Y","Z"].map(name => {
    const span = document.createElement("span");
    span.className = `axis-label axis-${name.toLowerCase()}`;
    span.textContent = name; span.hidden = true;
    labelLayer.append(span); return span;
  });

  function buf(data, target = gl.ARRAY_BUFFER) {
    const h = gl.createBuffer(); gl.bindBuffer(target, h);
    gl.bufferData(target, data, gl.STATIC_DRAW); return h;
  }
  function lineSet(segs) {
    const pos = new Float32Array(segs.length*6), col = new Uint8Array(segs.length*6);
    segs.forEach(([from,to,colour],i) => {
      pos.set(from,i*6); pos.set(to,i*6+3);
      col.set(colour,i*6); col.set(colour,i*6+3);
    });
    return { position: buf(pos), colour: buf(col), count: segs.length*2 };
  }
  function buildGrid(half) {
    const segs=[], steps=Math.round(half/options.gridMinor);
    const ratio=Math.max(1,Math.round(options.gridMajor/options.gridMinor));
    for (let i=-steps;i<=steps;i++) {
      const v=i*options.gridMinor, c=i%ratio===0?[84,95,108]:[40,47,56];
      segs.push([[v,-half,0],[v,half,0],c],[[-half,v,0],[half,v,0],c]);
    }
    return lineSet(segs);
  }
  function buildAxes(len) {
    const lines = lineSet(AXIS_COLOURS.map((c,ax)=>{ const t=[0,0,0];t[ax]=len;return[[0,0,0],t,c]; }));
    const per=240, pos=new Float32Array(3*per*3), col=new Uint8Array(3*per*3);
    for (let ax=0;ax<3;ax++) for (let i=0;i<per;i++) {
      const off=(ax*per+i)*3; pos[off+ax]=(len*i)/(per-1); col.set(AXIS_COLOURS[ax],off);
    }
    return { lines, dots: { position: buf(pos), colour: buf(col), count: 3*per } };
  }
  function delSet(s) { if(!s)return; for(const k of["position","colour"]) if(s[k])gl.deleteBuffer(s[k]); }
  function releaseBufs() {
    for(const h of Object.values(state.buffers)) gl.deleteBuffer(h);
    state.buffers={}; state.kind=null;
  }
  function frame(extents) {
    state.extents = extents.map(v=>Math.max(v,1e-3));
    const longest=Math.max(...state.extents);
    state.axisLength = longest*0.6;
    if(state.axes){delSet(state.axes.lines);delSet(state.axes.dots);}
    state.axes = buildAxes(state.axisLength);
    const half=Math.max(options.gridMajor,Math.ceil((longest*0.75)/options.gridMajor)*options.gridMajor);
    delSet(state.grid); state.grid=buildGrid(half);
    state.home={yaw:-1.1,pitch:0.38,distance:longest*2.4,target:[0,0,state.extents[2]/2]};
    reset();
  }
  function setMesh(mesh) {
    releaseBufs(); if(!mesh){requestDraw();return;}
    state.kind="mesh";
    state.buffers.position=buf(mesh.positions);
    if(mesh.normals) state.buffers.normal=buf(mesh.normals);
    state.buffers.colour=buf(mesh.colours);
    state.colourStride=Math.round(mesh.colours.length/mesh.count)||3;
    state.buffers.index=buf(mesh.indices,gl.ELEMENT_ARRAY_BUFFER);
    state.indexCount=mesh.indices.length;
    state.indexType=mesh.indices instanceof Uint32Array?gl.UNSIGNED_INT:gl.UNSIGNED_SHORT;
    state.count=mesh.count;
    if(state.indexType===gl.UNSIGNED_INT&&!uintIndices) state.kind=null;
    frame(mesh.extents);
  }
  function reset() {
    const h=state.home||{yaw:-1.1,pitch:0.38,distance:3,target:[0,0,0.5]};
    state.camera={yaw:h.yaw,pitch:h.pitch,distance:h.distance,target:h.target.slice()};
    requestDraw();
  }
  function eyePos() {
    const {yaw,pitch,distance,target}=state.camera;
    return [target[0]+distance*Math.cos(pitch)*Math.cos(yaw),
            target[1]+distance*Math.cos(pitch)*Math.sin(yaw),
            target[2]+distance*Math.sin(pitch)];
  }
  function bind(h,loc2,size,type,norm,stride) {
    if(loc2<0)return; gl.bindBuffer(gl.ARRAY_BUFFER,h);
    gl.enableVertexAttribArray(loc2); gl.vertexAttribPointer(loc2,size,type,norm,stride||0,0);
  }
  function constant(loc2,vals) {
    if(loc2<0)return; gl.disableVertexAttribArray(loc2); gl.vertexAttrib3f(loc2,vals[0],vals[1],vals[2]);
  }
  function drawLines(s,mode) {
    if(!s)return; bind(s.position,loc.position,3,gl.FLOAT,false);
    bind(s.colour,loc.color,3,gl.UNSIGNED_BYTE,true);
    constant(loc.normal,[0,0,1]); gl.uniform1f(loc.lit,0); gl.uniform1f(loc.flat,0);
    gl.drawArrays(mode,0,s.count);
  }
  function drawContent() {
    if(!state.kind)return;
    bind(state.buffers.position,loc.position,3,gl.FLOAT,false);
    bind(state.buffers.colour,loc.color,3,gl.UNSIGNED_BYTE,true,state.colourStride);
    gl.uniform1f(loc.flat,state.flat?1:0); gl.uniform3f(loc.flatColor,0.74,0.79,0.86);
    if(state.kind==="mesh"){
      if(state.buffers.normal) bind(state.buffers.normal,loc.normal,3,gl.FLOAT,false);
      else constant(loc.normal,[0,0,1]);
      gl.uniform1f(loc.lit,state.buffers.normal?1:0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,state.buffers.index);
      gl.drawElements(gl.TRIANGLES,state.indexCount,state.indexType,0);
    }
  }
  function placeLabels(mvp) {
    const w=canvas.clientWidth,h=canvas.clientHeight;
    axisLabels.forEach((span,ax)=>{
      const tip=[0,0,0]; tip[ax]=state.axisLength*1.08;
      const x=mvp[0]*tip[0]+mvp[4]*tip[1]+mvp[8]*tip[2]+mvp[12];
      const y=mvp[1]*tip[0]+mvp[5]*tip[1]+mvp[9]*tip[2]+mvp[13];
      const ww=mvp[3]*tip[0]+mvp[7]*tip[1]+mvp[11]*tip[2]+mvp[15];
      if(ww<=1e-6){span.hidden=true;return;}
      span.hidden=false;
      span.style.transform=`translate(${((x/ww)*0.5+0.5)*w}px,${(0.5-(y/ww)*0.5)*h}px)`;
    });
  }
  function draw() {
    state.pending=false;
    const dpr=window.devicePixelRatio||1;
    const width=Math.max(2,Math.floor(canvas.clientWidth*dpr));
    const height=Math.max(2,Math.floor(canvas.clientHeight*dpr));
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    gl.viewport(0,0,width,height);
    gl.clearColor(0.078,0.094,0.114,1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(program);
    const longest=Math.max(...state.extents);
    const proj=perspective(FOV,width/height,longest*0.01,longest*200);
    const mvp=multiply(proj,lookAt(eyePos(),state.camera.target,UP));
    gl.uniformMatrix4fv(loc.matrix,false,mvp);
    gl.uniform1f(loc.pointSize,1);
    gl.enable(gl.DEPTH_TEST);
    if(state.ground) drawLines(state.grid,gl.LINES);
    drawContent();
    gl.disable(gl.DEPTH_TEST);
    if(state.axes){ drawLines(state.axes.lines,gl.LINES); gl.uniform1f(loc.pointSize,2.5*dpr); drawLines(state.axes.dots,gl.POINTS); }
    placeLabels(mvp);
  }
  function requestDraw() { if(state.pending)return; state.pending=true; window.requestAnimationFrame(draw); }

  let drag=null;
  canvas.addEventListener("contextmenu",e=>e.preventDefault());
  canvas.addEventListener("mousedown",e=>{ drag={x:e.clientX,y:e.clientY,pan:e.button===2}; canvas.classList.add("dragging"); });
  window.addEventListener("mouseup",()=>{ drag=null; canvas.classList.remove("dragging"); });
  window.addEventListener("mousemove",e=>{
    if(!drag)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
    const cam=state.camera;
    if(drag.pan){
      const fwd=normalise(subtract(cam.target,eyePos()));
      const right=normalise(cross(fwd,UP)); const up=cross(right,fwd);
      const s=(cam.distance*1.1)/Math.max(1,canvas.clientHeight);
      for(let i=0;i<3;i++) cam.target[i]+=(-dx*right[i]+dy*up[i])*s;
    } else { cam.yaw-=dx*0.008; cam.pitch=Math.max(-1.45,Math.min(1.45,cam.pitch+dy*0.008)); }
    requestDraw();
  });
  canvas.addEventListener("wheel",e=>{
    e.preventDefault();
    const longest=Math.max(...state.extents);
    state.camera.distance=Math.max(longest*0.3,Math.min(longest*20,state.camera.distance*Math.exp(e.deltaY*0.0012)));
    requestDraw();
  },{passive:false});
  window.addEventListener("resize",requestDraw);
  frame(state.extents);
  return { setMesh, reset, requestDraw,
    setGround(v){state.ground=v;requestDraw();},
    setFlat(v){state.flat=v;requestDraw();},
  };
}

/* ── page ── */

const meshCanvas = $("meshCanvas");
const meshAxisLabels = $("meshAxisLabels");
let meshViewer = createViewer(meshCanvas, meshAxisLabels, { gridMinor: 0.1, gridMajor: 0.5 });
if (!meshViewer) $("fallback").hidden = false;

$("meshGround").addEventListener("change", e => meshViewer && meshViewer.setGround(e.target.checked));
$("meshFlat").addEventListener("change", e => meshViewer && meshViewer.setFlat(e.target.checked));
$("meshGround").checked = true;
$("meshFlat").checked = false;
$("meshFlatLabel").hidden = false;
document.querySelector("[data-reset='mesh']").addEventListener("click", () => meshViewer && meshViewer.reset());

// Stats
const byConf = {};
objects.forEach(o => { const c=o.physics.confidence||"?"; byConf[c]=(byConf[c]||0)+1; });
$("objectCount").textContent = objects.length;
$("countModel").textContent = byConf["model"]||0;
$("countClamped").textContent = byConf["model+clamped"]||0;
$("countFallback").textContent = byConf["fallback"]||0;

// Object list
const list = $("objectList");
let activeId = null;

function filterObjects() {
  const q = $("search").value.toLowerCase();
  const onlyKin = $("filterKinematic").checked;
  const onlyClamped = $("filterClamped").checked;
  const onlyDecomp = $("filterDecomp").checked;
  return objects.filter(o => {
    if (q && !o.object_id.includes(q) && !(o.label||"").toLowerCase().includes(q)) return false;
    if (onlyKin && !o.physics.kinematic) return false;
    if (onlyClamped && o.physics.confidence !== "model+clamped") return false;
    if (onlyDecomp && o.physics.collision_approximation !== "convexDecomposition") return false;
    return true;
  });
}

function buildList(filtered) {
  list.innerHTML = "";
  filtered.forEach(obj => {
    const li = document.createElement("li");
    if (obj.object_id === activeId) li.classList.add("active");
    const conf = obj.physics.confidence || "";
    let badges = "";
    if (conf === "model+clamped") badges += `<span class="badge badge-clamped">clamped</span>`;
    if (conf === "fallback") badges += `<span class="badge badge-fallback">fallback</span>`;
    if (obj.physics.kinematic) badges += `<span class="badge badge-kinematic">fixed</span>`;
    if (obj.physics.kinematic_reason) badges += `<span class="badge badge-forced">forced</span>`;
    li.innerHTML = `<div class="oid">${obj.object_id}</div><div class="lbl">${obj.label||"—"}${badges}</div>`;
    li.addEventListener("click", () => select(obj.object_id));
    list.appendChild(li);
  });
}

["search","filterKinematic","filterClamped","filterDecomp"].forEach(id => {
  const el = $(id);
  el.addEventListener("input", () => buildList(filterObjects()));
  el.addEventListener("change", () => buildList(filterObjects()));
});

function select(oid) {
  activeId = oid;
  buildList(filterObjects());
  const obj = objects.find(o => o.object_id === oid); if (!obj) return;

  $("detailHead").innerHTML =
    `<h2>${obj.label||"—"}</h2><span class="detail-id">${obj.object_id}</span>`;

  if (meshViewer) {
    const mesh = decodeObject(obj);
    meshViewer.setMesh(mesh);
  }

  // Keyframes
  const kfDiv = $("keyframes");
  kfDiv.innerHTML = "";
  obj.key_frames.forEach(kf => {
    const fig = document.createElement("figure");
    const img = document.createElement("img"); img.src = kf.image; img.alt = `seq ${kf.seq}`;
    const cap = document.createElement("figcaption");
    cap.textContent = `seq ${kf.seq}  ·  ${kf.role||""}  ·  ${(kf.visible_pixels||0).toLocaleString()} px`;
    fig.append(img, cap); kfDiv.appendChild(fig);
  });

  // Facts
  const p = obj.physics, s = obj.scale;
  const dims = obj.mesh.extents ? obj.mesh.extents.map(v=>v.toFixed(3)+"m").join(" × ") : "—";
  const confClass = {"model":"conf-model","model+clamped":"conf-clamped","fallback":"conf-fallback"}[p.confidence]||"";
  function fr(en, zh, val) {
    return `<div class="fact-row"><span class="key-en">${en}</span><div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px"><span class="key-zh">${zh}</span><span class="val">${val}</span></div></div>`;
  }
  const fmtVec = v => Array.isArray(v) ? v.map(x=>Number(x).toFixed(4)).join(", ") : "—";
  const fmtQuat = v => Array.isArray(v) ? v.map(x=>Number(x).toFixed(4)).join(", ") : "—";
  const fmtDecomp = d => {
    if (!d || typeof d !== "object") return "—";
    return Object.entries(d).map(([k,v])=>`${k}=${v}`).join(", ");
  };
  const kinLine = {
    "dynamic":   "自由刚体 (dynamic)",
    "kinematic": p.kinematic_reason
      ? `驱动刚体 (kinematic)${p.kinematic_reason ? " — " + p.kinematic_reason : ""}`
      : "驱动刚体 (kinematic)",
    "static":    p.kinematic_reason
      ? `静态碰撞体 (static) — ${p.kinematic_reason}`
      : "静态碰撞体 (static)",
  }[p.body_type] || (p.kinematic ? "固定安装 (kinematic)" : "自由刚体");
  $("facts").innerHTML = `
  <div class="fact-group"><h3>刚体</h3>
    ${fr("body_type","物理类型",kinLine)}
    ${fr("mass_kg","质量",p.mass_kg != null ? `${p.mass_kg} kg` : "—")}
    ${fr("rigid_body_enabled","刚体启用",p.rigid_body_enabled?"是":"否")}
    ${fr("center_of_mass","质心",fmtVec(p.center_of_mass))}
    ${fr("diagonal_inertia","主惯性矩",fmtVec(p.diagonal_inertia))}
    ${fr("principal_axes","惯性主轴 (wxyz)",fmtQuat(p.principal_axes))}
    ${fr("mass_distribution_assumption","质量分布假设",p.mass_distribution_assumption||"—")}
  </div>
  <div class="fact-group"><h3>碰撞</h3>
    ${fr("collision_enabled","碰撞启用",p.collision_enabled?"是":"否")}
    ${fr("collision_approximation","碰撞近似",p.collision_approximation||"—")}
    ${fr("collision_prim","碰撞体 prim",p.collision_prim||"—")}
    ${fr("visual_faces","视觉网格面数",p.visual_faces!=null?p.visual_faces.toLocaleString():"—")}
    ${fr("collision_faces","碰撞网格面数",p.collision_faces!=null?p.collision_faces.toLocaleString():"—")}
    ${fr("max_convex_hulls","最大凸包数",p.max_convex_hulls)}
    ${fr("voxel_resolution","体素分辨率",p.voxel_resolution)}
    ${fr("convex_decomposition_settings","凸包分解设置",fmtDecomp(p.convex_decomposition_settings))}
    ${fr("static_friction","静摩擦系数",p.static_friction)}
    ${fr("dynamic_friction","动摩擦系数",p.dynamic_friction)}
    ${fr("restitution","弹性恢复系数",p.restitution)}
    ${fr("solver_velocity_iterations","速度求解器迭代",p.solver_velocity_iterations!=null?p.solver_velocity_iterations:"PhysX 默认")}
    ${fr("solver_position_iterations","位置求解器迭代",p.solver_position_iterations!=null?p.solver_position_iterations:"PhysX 默认")}
  </div>
  <div class="fact-group"><h3>外观</h3>
    ${fr("roughness","粗糙度",p.roughness)}
    ${fr("metallic","金属度",p.metallic)}
    ${fr("opacity","不透明度",p.opacity)}
  </div>
  <div class="fact-group"><h3>尺度</h3>
    ${fr("scale_factor","缩放系数",s.scale_factor)}
    ${fr("height_m","高度",`${s.height_m} m`)}
    ${fr("footprint","长 × 宽",`${s.footprint_long_m} × ${s.footprint_short_m} m`)}
    ${fr("extents","缩放后尺寸",dims)}
    ${fr("on_floor","落地",s.on_floor===true?"是":s.on_floor===false?"否":"—")}
  </div>
  <div class="fact-group"><h3>尺度来源</h3>
    ${fr("source","来源",s.source||"—")}
    ${fr("mesh_scene_z_thin_frac","竖直薄度",s.mesh_scene_z_thin_frac)}
    ${fr("confidence","置信度",`<span class="${confClass}">${p.confidence||"—"}</span>`)}
  </div>`;
}

buildList(objects);
if (objects.length > 0) select(objects[0].object_id);

})();
