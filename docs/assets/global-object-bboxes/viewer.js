/* WebGL viewer adapted from harness_window01_v3/viewer.js.
 * Camera, pointer interaction, projection, matrix helpers and point shaders are copied
 * from that viewer. Bbox buffers, labels and object controls are the data adapter.
 * Original world coordinates and experiment records are never changed.
 */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const data = window.GLOBAL_OBJECT_BBOXES;
  const canvas = $('viewport');
  const overlay = document.createElement('canvas');
  overlay.className = 'bbox-label-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  canvas.after(overlay);
  const ctx = overlay.getContext('2d');
  let glState = null;
  // Normalized display-space distances, not meters.
  const MIN_CAMERA_DISTANCE = 0.005 / 3;
  const MAX_CAMERA_DISTANCE = 1000 * 3;
  let width = 1, height = 1, view = 'orbit', edgesOnScreen = [];
  // Screen-space triangle strips keep bbox edges visibly thick on all WebGL drivers.
  const BBOX_LINE_WIDTH = 3; // CSS pixels
  const FOCUSED_BBOX_LINE_WIDTH = 4;
  let linePositionBuffer, lineOtherBuffer, lineSideBuffer, lineColorBuffer, lineProgram, lineLocations;
  let instancePositionBuffer, instanceColorBuffer, instanceCount = 0;
  let visibleInstanceIds = [];
  let visibleInstanceColors = [];

  if (!data || !Array.isArray(data.objects)) {
    $('empty-state').hidden = false;
    $('empty-state').textContent = '未读取到包围盒数据。请确认 assets/global-object-bboxes/data.js 与本页面一起保存。';
    $('visible-count').textContent = '数据未加载';
    return;
  }
  let bboxMode = 'obb';
  const bounds = o => bboxMode === 'raw' ? (o.bbox_3d_raw || o.bbox_3d) : o.bbox_3d;
  const boxCorners = o => {
    if (bboxMode === 'obb' && o.bbox_3d_obb) return o.bbox_3d_obb.corners_xyz;
    const {min_xyz: lo, max_xyz: hi} = bounds(o);
    return Array.from({length:8}, (_,i) => [i&1?hi[0]:lo[0],i&2?hi[1]:lo[1],i&4?hi[2]:lo[2]]);
  };
  const modeName = () => ({obb:'直立方向框 OBB（已去离群点）',filtered:'轴对齐 AABB（已去离群点）',raw:'原始 AABB（未过滤）'}[bboxMode]);
  const cloudPoints = data.scene_cloud?.points || [];
  const instancePoints = data.instance_cloud?.points || [];
  const instanceObjectIds = data.instance_cloud?.object_ids || [];
  const instanceColors = data.instance_cloud?.colors || [];
  const numericId = o => { const m = String(o.object_id).match(/\d+/g); return m ? Number(m[m.length - 1]) : Infinity; };
  const objects = [...data.objects].sort((a, b) => numericId(a) - numericId(b) || String(a.object_id).localeCompare(String(b.object_id)));
  const name = o => `Object ${Number.isFinite(numericId(o)) ? String(numericId(o)).padStart(3, '0') : String(o.object_id)}`;
  const valid = o => o.bbox_3d && ['min_xyz', 'max_xyz'].every(k => Array.isArray(o.bbox_3d[k]) && o.bbox_3d[k].length === 3 && o.bbox_3d[k].every(Number.isFinite)) && o.bbox_3d.min_xyz.every((v, i) => v <= o.bbox_3d.max_xyz[i]);
  const usable = objects.filter(valid);
  const objectVisibility = new Map(objects.map(o => [o.object_id, { cloud: true, bbox: valid(o) }]));
  const instanceIndicesByObject = new Map();
  instanceObjectIds.forEach((raw, index) => {
    const id = Number(raw);
    if (!instanceIndicesByObject.has(id)) instanceIndicesByObject.set(id, []);
    instanceIndicesByObject.get(id).push(index);
  });
  let focused = usable[0]?.object_id ?? objects[0]?.object_id;
  const color = new Map(objects.map((o, i) => {
    const rgb = data.instance_cloud?.object_colors?.[String(numericId(o))];
    return [o.object_id, Array.isArray(rgb) ? `rgb(${rgb[0]} ${rgb[1]} ${rgb[2]})` : `hsl(${(150 + i * 137.508) % 360} 74% 69%)`];
  }));
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  usable.forEach(o => min.forEach((_, i) => { const box = o.bbox_3d_raw || o.bbox_3d; min[i] = Math.min(min[i], box.min_xyz[i]); max[i] = Math.max(max[i], box.max_xyz[i]); }));
  if (!usable.length) { min.fill(-1); max.fill(1); }
  cloudPoints.forEach(p => { if (p.length === 3 && p.every(Number.isFinite)) p.forEach((v, i) => { min[i] = Math.min(min[i], v); max[i] = Math.max(max[i], v); }); });
  instancePoints.forEach(p => { if (p.length === 3 && p.every(Number.isFinite)) p.forEach((v, i) => { min[i] = Math.min(min[i], v); max[i] = Math.max(max[i], v); }); });
  const sceneCenter = min.map((v, i) => (v + max[i]) / 2);
  const sceneRadius = Math.max(...max.map((v, i) => v - min[i]), 1);
  try { initWebgl(); } catch (error) {
    $('empty-state').hidden = false;
    $('empty-state').textContent = `3D 视窗初始化失败：${error.message}`;
    return;
  }
  glState.center = [...sceneCenter];
  glState.radius = sceneRadius;
  uploadScene();
  const unit = data.coordinate_system?.unit || 'm';
  const fmt = v => Math.abs(v) < 0.00005 ? '0.0000' : v.toFixed(4);
  const vector = a => a.map(fmt).join(', ');
  $('scene-summary').textContent = String(data.scene_id || 'first30');
  $('coordinate-tag').textContent = `${data.coordinate_system?.frame || 'GLOBAL'} · ${data.coordinate_system?.up_axis || 'Z'} ↑ · ${unit}`;
  $('object-count').textContent = objects.length;
  $('provenance').textContent = `Scene: ${data.scene_id ?? '—'}  /  State version: ${data.state_version ?? '—'}  /  Cutoff seq: ${data.cutoff_seq ?? '—'}  /  坐标顺序: X, Y, Z (${unit})`;
  $('show-scene').disabled = !cloudPoints.length;
  $('show-instances').disabled = !instancePoints.length;
  $('show-scene').checked = cloudPoints.length > 0;
  $('show-instances').checked = instancePoints.length > 0;
  $('cloud-count').textContent = `${cloudPoints.length.toLocaleString()} 个场景显示点 · ${instancePoints.length.toLocaleString()} 个实例点`;
  function rebuildInstanceBuffers() {
    const positions = [], colors = [], ids = [];
    objects.forEach(o => {
      if (!objectVisibility.get(o.object_id)?.cloud) return;
      const id = numericId(o);
      (instanceIndicesByObject.get(id) || []).forEach(i => {
        const point = instancePoints[i];
        if (!point) return;
        positions.push(...point);
        colors.push(...(instanceColors[i] || [220, 220, 220]));
        ids.push(id);
      });
    });
    visibleInstanceIds = ids;
    visibleInstanceColors = colors;
    instanceCount = ids.length;
    const gl = glState?.gl;
    if (gl && instancePositionBuffer && instanceColorBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, instancePositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, instanceColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.DYNAMIC_DRAW);
    }
  }

  $('filter-summary').textContent = '默认直立 OBB：先做 16 邻域统计过滤；若 8cm 连通半径下的主簇占比至少 75%，仅用主簇计算框，否则保留多簇。对至少 800 个点的较大物体，世界坐标轴两端的极端点裁剪已从各 0.5% 放宽为各 0.25%；KNN 密度阈值从中位数 + 3 MAD 放宽为中位数 + 4.5 MAD，减少误删真实结构。固定框的 Z 轴与世界 Z 轴一致；可靠的近竖直外表面以其水平法向约束框的水平朝向，无可靠竖直外表面时回退到 XY 主轴分析。水平外表面只约束高度，不提供水平朝向。通过单侧检验排除物体内部平面后，相应框面贴合支持该平面的表面点并保留噪声余量，其余范围由过滤后的物体点云确定；不按局部平面的水平轮廓裁剪整个物体。框只绕世界 Z 轴旋转，原始实例点云不变。过滤是几何启发式，仍可能遗漏真实分离部件；bbox 内也可能包含属于其它物体的空间重叠点。';
  const rows = new Map();
  objects.forEach(o => {
    const row = document.createElement('div'); row.className = 'object-row';
    const swatch = document.createElement('span'); swatch.className = 'swatch'; swatch.style.background = color.get(o.object_id);
    const button = document.createElement('button'); button.type = 'button'; button.className = 'object-main'; button.title = String(o.object_id);
    const text = document.createElement('span'); const title = document.createElement('span'); title.className = 'object-name'; title.textContent = name(o);
    const label = document.createElement('span'); label.className = 'object-label'; label.textContent = o.object_label || '未命名对象'; text.append(title, label); button.append(swatch, text);
    button.addEventListener('click', () => { focused = o.object_id; if (valid(o)) { objectVisibility.get(o.object_id).cloud = true; objectVisibility.get(o.object_id).bbox = true; $('show-instances').checked = true; $('show-bboxes').checked = true; } refresh(); });
    const count = document.createElement('span'); count.className = 'object-number'; count.textContent = Number(o.point_count || 0).toLocaleString(); count.title = '对象点数';
    const cloudLabel = document.createElement('label'); cloudLabel.className = 'object-toggle'; const cloud = document.createElement('input'); cloud.type = 'checkbox'; cloud.checked = objectVisibility.get(o.object_id).cloud; cloud.disabled = !instanceIndicesByObject.has(numericId(o)); cloud.setAttribute('aria-label', `${name(o)} 点云`); cloudLabel.append(cloud, '点云');
    const bboxLabel = document.createElement('label'); bboxLabel.className = 'object-toggle'; const bbox = document.createElement('input'); bbox.type = 'checkbox'; bbox.checked = objectVisibility.get(o.object_id).bbox; bbox.disabled = !valid(o); bbox.setAttribute('aria-label', `${name(o)} 3D 框`); bboxLabel.append(bbox, '框');
    cloud.addEventListener('change', () => { objectVisibility.get(o.object_id).cloud = cloud.checked; rebuildInstanceBuffers(); refresh(); });
    bbox.addEventListener('change', () => { objectVisibility.get(o.object_id).bbox = bbox.checked; refresh(); });
    row.append(button, count, cloudLabel, bboxLabel); $('object-list').append(row); rows.set(o.object_id, {row, cloud, bbox});
  });
  function details() {
    const o = objects.find(o => o.object_id === focused); const el = $('object-details'); el.replaceChildren();
    if (!o) { const p = document.createElement('p'); p.className = 'placeholder'; p.textContent = '当前没有可查看的对象。'; el.append(p); return; }
    const title = document.createElement('h3'); title.textContent = name(o); title.style.color = color.get(o.object_id);
    const label = document.createElement('p'); label.className = 'detail-label'; label.textContent = o.object_label || '未命名对象'; el.append(title, label);
    if (valid(o)) {
      const lo = bounds(o).min_xyz, hi = bounds(o).max_xyz;
      const table = document.createElement('table'); table.className = 'detail-table'; table.setAttribute('aria-label', '包围盒坐标，顺序为 X Y Z');
      const obb = bboxMode === 'obb' ? o.bbox_3d_obb : null;
      const fields = obb ? [['中心（世界 XYZ）',obb.center_xyz],['尺寸（局部 XYZ）',obb.size_xyz],...obb.axes_world.map((axis,i)=>[`局部 ${'XYZ'[i]} 轴（世界方向）`,axis])] : [['最小值',lo],['最大值',hi],['中心',lo.map((v,i)=>(v+hi[i])/2)],['尺寸',lo.map((v,i)=>hi[i]-v)]];
      fields.forEach(([k, v]) => { const tr=document.createElement('tr'), th=document.createElement('th'), td=document.createElement('td'); th.scope='row'; th.textContent=k; td.textContent=vector(v); tr.append(th,td); table.append(tr); }); el.append(table);
    } else { const p=document.createElement('p'); p.className='placeholder'; p.textContent='该对象没有有效点云包围盒。'; el.append(p); }
    const foot = document.createElement('p'); foot.className='detail-foot'; foot.textContent=`中心/坐标单位：${unit} · 尺寸沿当前框的三个轴；方向为单位向量；直立 OBB 的 Z 轴固定为世界 Z 轴\n${Number(o.point_count || 0).toLocaleString()} 个点 · ${o.observation_count ?? 0} 次观测`; foot.style.whiteSpace='pre-line'; el.append(foot);
    const raw = o.bbox_3d_raw || o.bbox_3d, filtered = o.bbox_3d;
    if (raw && filtered) {
      const volume = box => box.min_xyz.reduce((v, x, i) => v * Math.max(0, box.max_xyz[i] - x), 1);
      const rawVolume = volume(raw), filteredVolume = volume(filtered);
      const stats = document.createElement('p'); stats.className = 'filter-stats';
      const f = o.filtering || {};
      const obbVolume = o.bbox_3d_obb?.size_xyz.reduce((a,b)=>a*b,1);
      stats.textContent = `当前：${modeName()}\n保留 ${Number(f.retained_points ?? o.point_count).toLocaleString()} 点 · 排除 ${Number(f.removed_points || 0).toLocaleString()} 点\n原始 AABB：${rawVolume.toFixed(4)} m³\n过滤后 AABB：${filteredVolume.toFixed(4)} m³${obbVolume === undefined ? '' : `\n同批过滤点直立 OBB：${obbVolume.toFixed(4)} m³`}`;
      el.append(stats);
      if (o.filtering) { const detail = document.createElement('details'), summary = document.createElement('summary'), pre = document.createElement('pre'); detail.className = 'filter-record'; summary.textContent = '查看过滤参数'; pre.textContent = JSON.stringify(o.filtering, null, 2); detail.append(summary, pre); el.append(detail); }
    }
  }
  function refresh() { rows.forEach(({row,cloud,bbox}, id) => { row.classList.toggle('focused', id===focused); cloud.checked=objectVisibility.get(id)?.cloud ?? false; bbox.checked=objectVisibility.get(id)?.bbox ?? false; }); details(); draw(); }
  const edgeIndices = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  rebuildInstanceBuffers();
  function project(p, matrix) {
    const clip = [0,0,0,0].map((_, row) => matrix[row]*p[0] + matrix[4+row]*p[1] + matrix[8+row]*p[2] + matrix[12+row]);
    if (clip[3] <= 0 || clip[2] < -clip[3] || clip[2] > clip[3]) return null;
    return [(clip[0]/clip[3]+1)*width/2, (1-clip[1]/clip[3])*height/2];
  }
  function uploadScene() {
    const gl = glState.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, glState.bgPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(cloudPoints.flat()), gl.STATIC_DRAW);
    const colors = cloudPoints.flatMap((_, i) => data.scene_cloud?.colors?.[i] || [130,145,160]);
    gl.bindBuffer(gl.ARRAY_BUFFER, glState.bgColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(colors), gl.STATIC_DRAW);
    glState.bgCount = cloudPoints.length;
    instancePositionBuffer = gl.createBuffer(); instanceColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instancePositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(instancePoints.flat()), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(instanceColors.flat()), gl.STATIC_DRAW);
    instanceCount = instancePoints.length;
    linePositionBuffer = gl.createBuffer(); lineOtherBuffer = gl.createBuffer();
    lineSideBuffer = gl.createBuffer(); lineColorBuffer = gl.createBuffer();
    lineProgram = createProgram(gl, `attribute vec3 a_position; attribute vec3 a_other;
      attribute float a_side; attribute vec3 a_color;
      uniform mat4 u_matrix; uniform vec2 u_viewport; varying vec3 v_color;
      void main(){
        vec4 p=u_matrix*vec4(a_position,1.0), q=u_matrix*vec4(a_other,1.0);
        // Clip endpoints at the near plane before perspective division.
        float dp=p.z+p.w, dq=q.z+q.w;
        if(dp<0.0 && dq>=0.0) p=mix(p,q,dp/(dp-dq));
        else if(dq<0.0 && dp>=0.0) q=mix(q,p,dq/(dq-dp));
        vec2 direction=(q.xy/max(q.w,0.000001)-p.xy/max(p.w,0.000001))*u_viewport;
        vec2 normal=vec2(-direction.y,direction.x)/max(length(direction),0.000001);
        p.xy+=normal*a_side/u_viewport*p.w;
        gl_Position=p;v_color=a_color/255.0;
      }`,
      `precision mediump float; varying vec3 v_color;
      void main(){gl_FragColor=vec4(v_color,1.0);}`);
    lineLocations = { position: gl.getAttribLocation(lineProgram,'a_position'), other: gl.getAttribLocation(lineProgram,'a_other'), side: gl.getAttribLocation(lineProgram,'a_side'), color: gl.getAttribLocation(lineProgram,'a_color'), matrix: gl.getUniformLocation(lineProgram,'u_matrix'), viewport: gl.getUniformLocation(lineProgram,'u_viewport') };
  }
  function objectRgb(o) {
    // Parse the same CSS palette used by the object list.
    ctx.fillStyle = color.get(o.object_id);
    const normalized = ctx.fillStyle;
    if (normalized.startsWith('#')) return normalized.slice(1).match(/../g).map(v=>parseInt(v,16));
    return normalized.match(/[\d.]+/g).slice(0,3).map(Number);
  }
  function draw() { renderCloud(); }
  function renderCloud() {
    if (!glState) return;
    const gl = glState.gl;
    resizeCanvasToDisplaySize(canvas);
    width=canvas.clientWidth; height=canvas.clientHeight;
    if (overlay.width!==canvas.width || overlay.height!==canvas.height) {overlay.width=canvas.width;overlay.height=canvas.height;}
    ctx.setTransform(canvas.width/width,0,0,canvas.height/height,0,0);
    ctx.clearRect(0,0,width,height); edgesOnScreen=[];
    gl.viewport(0,0,canvas.width,canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    const matrix=computeMatrix(canvas.width/canvas.height);
    gl.useProgram(glState.program); gl.uniformMatrix4fv(glState.matrixLocation,false,matrix);
    if ($('show-scene').checked && glState.bgCount) {
      // Brightness does not affect geometry or camera state.
      gl.uniform1f(glState.brightnessLocation,Number($('cloud-opacity').value)/100);
      drawBuffers(glState.bgPositionBuffer,glState.bgColorBuffer,glState.bgCount,1.0);
    }
    if ($('show-instances').checked && instanceCount) {
      gl.uniform1f(glState.brightnessLocation, 1.15);
      drawBuffers(instancePositionBuffer, instanceColorBuffer, instanceCount, 2.2);
    }
    const shown=usable.filter(o=>$('show-bboxes').checked && objectVisibility.get(o.object_id)?.bbox);
    const positions=[], otherPositions=[], sides=[], colors=[];
    shown.forEach(o=>{
      const corners=boxCorners(o);
      const projected=corners.map(p=>project(p,matrix));
      const rgb=objectRgb(o);
      edgeIndices.forEach(([a,b])=>{
        const lineWidth=o.object_id===focused?FOCUSED_BBOX_LINE_WIDTH:BBOX_LINE_WIDTH;
        // Each edge is two triangles. Reversing the endpoint also reverses its normal.
        [[a,b,-1],[a,b,1],[b,a,1],[b,a,1],[a,b,1],[b,a,-1]].forEach(([from,to,side])=>{
          positions.push(...corners[from]);otherPositions.push(...corners[to]);
          sides.push(side*lineWidth);colors.push(...rgb);
        });
        if(projected[a]&&projected[b])edgesOnScreen.push({a:projected[a],b:projected[b],id:o.object_id});
      });
      if (o.object_id!==focused && !$('show-labels').checked && !$('show-category-labels').checked) return;
      const visible=projected.filter(Boolean); if(!visible.length)return;
      const anchor=visible.reduce((a,b)=>a[1]<b[1]?a:b);
      if(anchor[0]<0||anchor[0]>width||anchor[1]<0||anchor[1]>height)return;
      const showId = o.object_id===focused || $('show-labels').checked;
      const text = [showId ? name(o) : '', $('show-category-labels').checked ? (o.object_label || '未命名对象') : ''].filter(Boolean).join(' · ');ctx.font='600 12px sans-serif';const tw=ctx.measureText(text).width;
      const x=Math.max(6,Math.min(width-tw-14,anchor[0]+5)),y=Math.max(68,Math.min(height-44,anchor[1]-12));
      ctx.fillStyle='#0b121de0';ctx.fillRect(x-5,y-13,tw+10,21);ctx.fillStyle=color.get(o.object_id);ctx.fillText(text,x,y+2);
    });
    gl.useProgram(lineProgram);gl.uniformMatrix4fv(lineLocations.matrix,false,matrix);
    gl.uniform2f(lineLocations.viewport,width,height);
    gl.bindBuffer(gl.ARRAY_BUFFER,linePositionBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(positions),gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(lineLocations.position);gl.vertexAttribPointer(lineLocations.position,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,lineOtherBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(otherPositions),gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(lineLocations.other);gl.vertexAttribPointer(lineLocations.other,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,lineSideBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(sides),gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(lineLocations.side);gl.vertexAttribPointer(lineLocations.side,1,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,lineColorBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Uint8Array(colors),gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(lineLocations.color);gl.vertexAttribPointer(lineLocations.color,3,gl.UNSIGNED_BYTE,false,0,0);
    gl.drawArrays(gl.TRIANGLES,0,positions.length/3);
    $('visible-count').textContent=`显示 ${shown.length} / ${usable.length} 个有效 Object · 点云 ${objects.filter(o=>objectVisibility.get(o.object_id)?.cloud).length} / ${objects.length}`;
    $('empty-state').hidden=shown.length>0||($('show-scene').checked&&cloudPoints.length>0);
    $('empty-state').textContent='尚未选择对象。点击右侧「全选」显示全部包围盒。';
  }
  document.querySelectorAll('[data-bbox]').forEach(button => button.addEventListener('click', () => { bboxMode = button.dataset.bbox; document.querySelectorAll('[data-bbox]').forEach(b => b.classList.toggle('active', b.dataset.bbox === bboxMode)); refresh(); }));
  $('show-scene').addEventListener('change', draw);
  $('show-instances').addEventListener('change', draw);
  $('show-bboxes').addEventListener('change', draw);
  $('show-labels').addEventListener('change', draw);
  $('show-category-labels').addEventListener('change', draw);
  $('cloud-opacity').addEventListener('input', draw);
  function setView(v) {
    view=v;
    glState.rotationX=v==='top'?0:v==='front'?-Math.PI/2:-0.35;
    glState.rotationY=v==='orbit'?0.45:0;
    document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
    draw();
  }
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
  function resetView(){glState.center=[...sceneCenter];glState.radius=sceneRadius;glState.distance=3.2;glState.panX=0;glState.panY=0;setView('orbit');}
  $('reset-view').addEventListener('click',resetView);
  $('focus-objects').addEventListener('click',()=>{
    const active=usable.filter(o=>objectVisibility.get(o.object_id)?.bbox);if(!active.length)return;
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    active.forEach(o=>boxCorners(o).forEach(p=>p.forEach((v,i)=>{lo[i]=Math.min(lo[i],v);hi[i]=Math.max(hi[i],v);})));
    glState.center=lo.map((v,i)=>(v+hi[i])/2);glState.radius=Math.max(...hi.map((v,i)=>v-lo[i]),0.1);
    glState.distance=3.2;glState.panX=0;glState.panY=0;draw();
  });
  $('select-all').addEventListener('click',()=>{objects.forEach(o=>{objectVisibility.get(o.object_id).cloud=true;objectVisibility.get(o.object_id).bbox=valid(o);});rebuildInstanceBuffers();refresh();});
  $('select-none').addEventListener('click',()=>{objects.forEach(o=>{objectVisibility.get(o.object_id).cloud=false;objectVisibility.get(o.object_id).bbox=false;});rebuildInstanceBuffers();refresh();});
  // Picking is separate from the copied camera handlers: it never alters drag behavior.
  let clickStart=null;
  canvas.addEventListener('pointerdown',e=>{if(e.button===0)clickStart={x:e.clientX,y:e.clientY,id:e.pointerId,moved:false};});
  canvas.addEventListener('pointermove',e=>{if(clickStart&&e.pointerId===clickStart.id&&Math.hypot(e.clientX-clickStart.x,e.clientY-clickStart.y)>4)clickStart.moved=true;});
  canvas.addEventListener('pointerup',e=>{
    if(!clickStart||e.pointerId!==clickStart.id)return;
    const click=clickStart;clickStart=null;if(click.moved)return;
    const rect=canvas.getBoundingClientRect(),p=[e.clientX-rect.left,e.clientY-rect.top];let best=null,closest=10;
    edgesOnScreen.forEach(edge=>{const dx=edge.b[0]-edge.a[0],dy=edge.b[1]-edge.a[1],den=dx*dx+dy*dy;
      const t=den?Math.max(0,Math.min(1,((p[0]-edge.a[0])*dx+(p[1]-edge.a[1])*dy)/den)):0;
      const d=Math.hypot(p[0]-edge.a[0]-t*dx,p[1]-edge.a[1]-t*dy);if(d<closest){best=edge.id;closest=d;}});
    if(best!==null){focused=best;refresh();rows.get(best)?.row.scrollIntoView({block:'nearest'});}
  });
  canvas.addEventListener('pointercancel',()=>{clickStart=null;});
  canvas.addEventListener('keydown',e=>{
    if(e.key==='Home'){e.preventDefault();resetView();}
    else if(e.key==='+'||e.key==='='){e.preventDefault();glState.distance=Math.max(MIN_CAMERA_DISTANCE,glState.distance/1.15);draw();}
    else if(e.key==='-'){e.preventDefault();glState.distance=Math.min(MAX_CAMERA_DISTANCE,glState.distance*1.15);draw();}
  });
  new ResizeObserver(draw).observe(canvas);
  window.addEventListener('resize',draw);
  refresh();

function initWebgl() {
  const canvas = $("viewport");
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
    uniform float u_brightness;
    void main() {
      vec2 c = gl_PointCoord - vec2(0.5);
      if (dot(c, c) > 0.25) discard;
      gl_FragColor = vec4(v_color * u_brightness, 0.95);
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
    brightnessLocation: gl.getUniformLocation(program, "u_brightness"),
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
    glState.distance = Math.max(MIN_CAMERA_DISTANCE, Math.min(MAX_CAMERA_DISTANCE, glState.distance));
    renderCloud();
  }, { passive: false });
}

function resizeCanvasToDisplaySize(canvas) {
  const width = Math.max(2, Math.floor(canvas.clientWidth * window.devicePixelRatio));
  const height = Math.max(2, Math.floor(canvas.clientHeight * window.devicePixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
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
  const projection = perspective(Math.PI / 4, aspect, Math.min(0.01, glState.distance * 0.02), Math.max(500, glState.distance * 2 + 10));
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


})();
