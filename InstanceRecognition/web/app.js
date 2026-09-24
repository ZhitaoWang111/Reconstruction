const elements = {
  scene: document.querySelector('#scene-select'),
  frame: document.querySelector('#frame-input'),
  slider: document.querySelector('#frame-slider'),
  previous: document.querySelector('#previous-frame'),
  next: document.querySelector('#next-frame'),
  load: document.querySelector('#load-frame'),
  infer: document.querySelector('#run-inference'),
  summary: document.querySelector('#dataset-summary'),
  statusIndicator: document.querySelector('#status-indicator'),
  statusTitle: document.querySelector('#status-title'),
  statusDetail: document.querySelector('#status-detail'),
  sourceMeta: document.querySelector('#source-meta'),
  badges: document.querySelector('#result-badges'),
  details: document.querySelector('#details-section'),
  table: document.querySelector('#instance-table'),
  modelService: document.querySelector('#model-service-state'),
};

const images = {
  source: document.querySelector('#source-image'),
  yoloe: document.querySelector('#yoloe-image'),
  sam31: document.querySelector('#sam31-image'),
};

const placeholders = {
  source: document.querySelector('#source-placeholder'),
  yoloe: document.querySelector('#yoloe-placeholder'),
  sam31: document.querySelector('#sam31-placeholder'),
};

let scenes = new Map();
let timer = null;
let modelStatusTimer = null;
let heartbeatTimer = null;
let leaseOpen = false;
let recognitionMethod = 'gpt_sam31';
const clientId = window.crypto?.randomUUID?.()
  || `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;

function selectedFrame() {
  return Math.max(0, Number.parseInt(elements.frame.value || '0', 10));
}

function selectedScene() {
  return scenes.get(elements.scene.value);
}

function setStatus(kind, title, detail) {
  elements.statusIndicator.className = `status-indicator ${kind}`;
  elements.statusTitle.textContent = title;
  elements.statusDetail.textContent = detail;
}

function showImage(key, url) {
  images[key].src = `${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`;
  images[key].classList.add('visible');
  placeholders[key].style.display = 'none';
}

function clearOutputs() {
  document.querySelector('#gpt-categories').replaceChildren();
  document.querySelector('#gpt-categories').classList.add('hidden');
  for (const key of ['yoloe', 'sam31']) {
    images[key].removeAttribute('src');
    images[key].classList.remove('visible');
    placeholders[key].style.display = '';
  }
  elements.badges.replaceChildren();
  elements.details.classList.add('hidden');
}

function syncFrameControls(value) {
  const scene = selectedScene();
  if (!scene) return;
  const frame = Math.min(Math.max(0, Number(value)), scene.frame_count - 1);
  elements.frame.value = frame;
  elements.slider.value = frame;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

function renderModelStatus(status) {
  const workers=Object.values(status.workers || {});
  elements.modelService.className='model-service-state';
  if(workers.length && workers.every(w=>w.loaded)){
    elements.modelService.classList.add('loaded');
    elements.modelService.textContent=recognitionMethod==='gpt_sam31'?'SAM 模型已驻留 · GPT 按需调用':'YOLOE / SAM 模型已驻留';
  } else if(workers.some(w=>w.running || w.loading)){
    elements.modelService.classList.add('loading');elements.modelService.textContent='模型预热中…';
  } else elements.modelService.textContent='模型未驻留';
}

async function refreshModelStatus() {
  try {
    renderModelStatus(await requestJson('/api/models/status'));
  } catch (_) {
    elements.modelService.textContent = '模型服务不可用';
  }
}

async function openClientLease() {
  if (leaseOpen) return;
  leaseOpen = true;
  try {
    const status = await requestJson('/api/client/open', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({client_id: clientId}),
    });
    renderModelStatus(status);
  } catch (error) {
    leaseOpen = false;
    elements.modelService.textContent = `模型服务连接失败：${error.message}`;
  }
}

function closeClientLease() {
  if (!leaseOpen) return;
  leaseOpen = false;
  const body = new Blob([JSON.stringify({client_id: clientId})], {
    type: 'application/json',
  });
  navigator.sendBeacon('/api/client/close', body);
}

async function loadFrame({checkCache = true} = {}) {
  const scene = selectedScene();
  if (!scene) return;
  syncFrameControls(selectedFrame());
  const frame = selectedFrame();
  clearOutputs();
  showImage('source', `/api/frame/${scene.scene_id}/${frame}.jpg`);
  elements.sourceMeta.textContent = `${scene.scene_id} · frame ${frame} / ${scene.frame_count - 1} · ${scene.color_width}×${scene.color_height}`;
  setStatus('ready', 'RGB 帧已加载', `${scene.scene_id} 的第 ${frame} 帧`);
  if (!checkCache) return;
  try {
    const cached = await requestJson(`/api/run/${scene.scene_id}/${frame}`);
    if (cached.available) renderResult(cached);
  } catch (error) {
    setStatus('error', '缓存读取失败', error.message);
  }
}

function addBadge(text) {
  const badge = document.createElement('span');
  badge.textContent = text;
  elements.badges.appendChild(badge);
}

function renderResult(result) {
  showImage('source', result.input_image);
  const isGpt=!!result.gpt;
  if (isGpt) {
    const list=document.querySelector('#gpt-categories');list.replaceChildren();list.classList.remove('hidden');
    images.yoloe.removeAttribute('src');images.yoloe.classList.remove('visible');placeholders.yoloe.style.display='none';
    document.querySelector('#label-stage-title').textContent=result.gpt.model.replace(/^openai\//,'');
    for(const item of result.gpt.categories){
      const card=document.createElement('article');const title=document.createElement('strong');
      title.textContent=item.category+' · '+(item.certainty==='clear'?'明确':'不确定');
      const evidence=document.createElement('p');evidence.textContent=item.evidence+(item.ambiguity?' / '+item.ambiguity:'');
      card.append(title,evidence);list.append(card);
    }
    for(const region of result.gpt.unresolved_regions){const text=document.createElement('p');text.textContent='未识别：'+region.location+' / '+region.reason;list.append(text);}
  } else showImage('yoloe', result.yoloe.image);
  showImage('sam31', result.sam31.image);
  elements.badges.replaceChildren();
  addBadge(`${result.sam31.instance_count} 个实例`);
  addBadge(`${isGpt?result.gpt.categories.length:result.yoloe.unique_classes.length} 类标签`);
  addBadge(`${result.wall_seconds.toFixed(1)} s`);
  if (result.cached) addBadge('已缓存');

  elements.table.replaceChildren();
  for (const item of result.sam31.instances) {
    const row = document.createElement('tr');
    const prompt = item.sam_prompt_type === 'text' ? item.sam_prompt : (item.sam_prompt_type === 'positive_box' ? '正框' : '中心点回退');
    const values = [
      item.instance_id,
      item.class_name,
      isGpt ? `SAM ${item.sam_score.toFixed(3)}` : `YOLOE ${item.yoloe_confidence.toFixed(3)}`,
      prompt,
      isGpt ? (item.gpt_certainty==='clear'?'明确':'不确定') : item.prompt_box_iou.toFixed(3),
      item.mask_pixels.toLocaleString(),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement('td');
      if (index === 3) {
        const pill = document.createElement('span');
        pill.className = 'prompt-pill';
        pill.textContent = value;
        cell.appendChild(pill);
      } else {
        cell.textContent = value;
      }
      row.appendChild(cell);
    });
    elements.table.appendChild(row);
  }
  document.querySelector('#yoloe-json').href = isGpt ? result.gpt.manifest : result.yoloe.manifest;
  document.querySelector('#sam-json').href = result.sam31.manifest;
  document.querySelector('#instance-map').href = result.sam31.instance_map;
  elements.details.classList.remove('hidden');
  const cacheText = result.cached ? '读取已有结果' : '两阶段推理完成';
  setStatus('ready', cacheText, isGpt ? `${result.gpt.categories.length} 类提示，${result.sam31.instance_count} 个 mask，${result.sam31.empty_prompt_count} 类未分出 mask` : `${result.yoloe.instance_count} 个检测均已获得 SAM3.1 mask`);
}

async function runInference() {
  const scene = selectedScene();
  if (!scene) return;
  const frame = selectedFrame();
  elements.infer.disabled = true;
  elements.load.disabled = true;
  const started = Date.now();
  setStatus('running', '正在运行两阶段推理', recognitionMethod==='gpt_sam31'?'正在调用 GPT 类别识别…':'正在运行 YOLOE 检测…');
  timer = window.setInterval(() => {
    const elapsed = Math.round((Date.now() - started) / 1000);
    setStatus('running', '正在运行两阶段推理', `${recognitionMethod==='gpt_sam31'?'GPT':'YOLOE'} → SAM3.1 · 已等待 ${elapsed} 秒`);
  }, 1000);
  try {
    const result = await requestJson('/api/infer', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({scene_id: scene.scene_id, frame_index: frame}),
    });
    renderResult(result);
  } catch (error) {
    setStatus('error', '推理失败', error.message);
  } finally {
    window.clearInterval(timer);
    elements.infer.disabled = false;
    elements.load.disabled = false;
  }
}

async function initialize() {
  try {
    recognitionMethod=(await requestJson('/api/health')).method;
    if(recognitionMethod==='yoloe_sam31'){
      document.querySelector('.lede').textContent='选择本地 ScanNet RGB 帧，由 YOLOE-26 检测类别与框，再由 SAM3.1 生成 mask。';
      document.querySelector('.pipeline-mark strong').textContent='YOLOE-26';
      document.querySelector('.pipeline-mark div span').textContent='类别与检测框';
      document.querySelector('#label-stage-title').textContent='YOLOE-26';
      document.querySelector('#label-stage-description').textContent='类别、置信度与检测框';
      document.querySelector('.featured .card-title p').textContent='由 YOLOE 检测框提示的实例 mask';
      document.querySelector('#run-inference').lastChild.textContent='启动 YOLOE + SAM3.1';
      document.querySelector('thead th:nth-child(5)').textContent='框 IoU';
      document.querySelector('.boundary-note').textContent='SAM3.1 只接收 YOLOE 检测框，不消费 YOLOE 原生 mask。';
    }
  } catch(error) {
    setStatus('error','配置读取失败',error.message);
    return;
  }
  await openClientLease();
  modelStatusTimer = window.setInterval(refreshModelStatus, 1000);
  heartbeatTimer = window.setInterval(async () => {
    if (!leaseOpen) return;
    try {
      renderModelStatus(await requestJson('/api/client/heartbeat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({client_id: clientId}),
      }));
    } catch (_) {
      // The next heartbeat/status poll retries automatically.
    }
  }, 5000);
  try {
    const data = await requestJson('/api/scenes');
    scenes = new Map(data.scenes.map(scene => [scene.scene_id, scene]));
    for (const scene of data.scenes) {
      const option = document.createElement('option');
      option.value = scene.scene_id;
      option.textContent = `${scene.scene_id} · ${scene.frame_count.toLocaleString()} 帧`;
      elements.scene.appendChild(option);
    }
    const totalFrames = data.scenes.reduce((sum, scene) => sum + scene.frame_count, 0);
    elements.summary.textContent = `${data.count} 个本地场景 · ${totalFrames.toLocaleString()} 个可访问 RGB 帧`;
    const query = new URLSearchParams(window.location.search);
    const requestedScene = query.get('scene');
    if (requestedScene && scenes.has(requestedScene)) {
      elements.scene.value = requestedScene;
    }
    const scene = selectedScene();
    elements.frame.max = scene.frame_count - 1;
    elements.slider.max = scene.frame_count - 1;
    syncFrameControls(Number.parseInt(query.get('frame') || '0', 10));
    loadFrame();
  } catch (error) {
    setStatus('error', 'ScanNet 数据读取失败', error.message);
    elements.summary.textContent = '数据集不可用';
  }
}

function updateScene() {
  const scene = selectedScene();
  if (!scene) return;
  elements.frame.max = scene.frame_count - 1;
  elements.slider.max = scene.frame_count - 1;
  syncFrameControls(0);
  loadFrame();
}

elements.scene.addEventListener('change', updateScene);
elements.frame.addEventListener('change', () => { syncFrameControls(selectedFrame()); loadFrame(); });
elements.slider.addEventListener('input', event => syncFrameControls(event.target.value));
elements.slider.addEventListener('change', () => loadFrame());
elements.previous.addEventListener('click', () => { syncFrameControls(selectedFrame() - 1); loadFrame(); });
elements.next.addEventListener('click', () => { syncFrameControls(selectedFrame() + 1); loadFrame(); });
elements.load.addEventListener('click', () => loadFrame());
elements.infer.addEventListener('click', runInference);
window.addEventListener('pagehide', closeClientLease);
window.addEventListener('pageshow', event => {
  if (event.persisted) openClientLease();
});

initialize();
