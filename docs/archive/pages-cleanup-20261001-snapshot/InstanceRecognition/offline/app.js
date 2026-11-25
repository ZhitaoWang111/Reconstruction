const data = window.OFFLINE_RESULTS_INDEX;

const elements = {
  scene: document.querySelector('#scene-select'),
  frame: document.querySelector('#frame-select'),
  slider: document.querySelector('#frame-slider'),
  previous: document.querySelector('#previous-frame'),
  next: document.querySelector('#next-frame'),
  position: document.querySelector('#frame-position'),
  range: document.querySelector('#raw-frame-range'),
  nearby: document.querySelector('#nearby-frames'),
  badges: document.querySelector('#frame-badges'),
  episodeMeta: document.querySelector('#episode-meta'),
  episodeTable: document.querySelector('#episode-table'),
  confidence: document.querySelector('#confidence-threshold'),
  confidenceValue: document.querySelector('#confidence-value'),
  minimumFrames: document.querySelector('#minimum-frames'),
  samViewLabel: document.querySelector('#sam-view-label'),
  overlay: document.querySelector('#show-overlay'),
  idMap: document.querySelector('#show-id-map'),
};

const images = {
  source: document.querySelector('#source-image'),
  yoloe: document.querySelector('#yoloe-image'),
  sam: document.querySelector('#sam-image'),
};

let scenePosition = 0;
let framePosition = 0;
let samView = 'overlay';
let confidenceThreshold = 0.5;

function currentScene() {
  return data.scenes[scenePosition];
}

function currentFrame() {
  return currentScene().frames[framePosition];
}

function formatSeconds(value) {
  return `${Number(value).toFixed(2)} s`;
}

function setQuery() {
  const frame = currentFrame();
  const query = `?scene=${encodeURIComponent(currentScene().scene_id)}&frame=${frame.frame_index}&confidence=${confidenceThreshold.toFixed(2)}`;
  window.history.replaceState(null, '', query);
}

function addBadge(text, accent = false) {
  const badge = document.createElement('span');
  badge.textContent = text;
  if (accent) badge.classList.add('accent');
  elements.badges.appendChild(badge);
}

function renderNearbyFrames() {
  elements.nearby.replaceChildren();
  const scene = currentScene();
  const start = Math.max(0, Math.min(framePosition - 3, scene.frames.length - 7));
  const end = Math.min(scene.frames.length, start + 7);
  for (let index = start; index < end; index += 1) {
    const button = document.createElement('button');
    button.textContent = scene.frames[index].frame_index;
    button.className = index === framePosition ? 'active' : '';
    button.addEventListener('click', () => setFrame(index));
    elements.nearby.appendChild(button);
  }
}

function renderEpisodeSummary() {
  const scene = currentScene();
  const minimumFrames = Math.max(1, Number.parseInt(elements.minimumFrames.value || '1', 10));
  elements.minimumFrames.value = String(Math.min(minimumFrames, scene.frame_count));
  elements.minimumFrames.max = String(scene.frame_count);
  const stats = computeEpisodeStats(
    scene,
    confidenceThreshold,
    Number(elements.minimumFrames.value),
  );
  elements.episodeMeta.textContent = `${scene.scene_id} · ${scene.frame_count} 个已处理帧 · 置信度 ≥ ${confidenceThreshold.toFixed(2)}`;
  document.querySelector('#episode-class-count').textContent = stats.class_count.toLocaleString();
  document.querySelector('#episode-detection-count').textContent = stats.instance_count.toLocaleString();
  document.querySelector('#episode-detected-frames').textContent = `${stats.detected_frames} / ${scene.frame_count}`;
  elements.episodeTable.replaceChildren();

  for (const item of stats.classes) {
    const row = document.createElement('tr');
    const values = [
      item.class_name,
      `${item.detected_frames} / ${scene.frame_count}`,
      `${(item.frame_ratio * 100).toFixed(1)}%`,
      item.instance_count.toLocaleString(),
      item.mean_confidence.toFixed(3),
      item.maximum_confidence.toFixed(3),
    ];
    for (const value of values) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
    }
    const frameCell = document.createElement('td');
    const button = document.createElement('button');
    button.className = 'representative-frame';
    button.textContent = `frame ${item.representative_frame}`;
    button.title = '跳到该类别置信度最高的帧';
    button.addEventListener('click', () => {
      setFrame(item.representative_position);
      document.querySelector('#viewer-title').scrollIntoView({behavior: 'smooth'});
    });
    frameCell.appendChild(button);
    row.appendChild(frameCell);
    elements.episodeTable.appendChild(row);
  }

  if (!stats.classes.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 7;
    cell.className = 'empty-result';
    cell.textContent = '当前阈值下没有满足条件的类别。';
    row.appendChild(cell);
    elements.episodeTable.appendChild(row);
  }
}

function renderFrame() {
  const scene = currentScene();
  const frame = currentFrame();
  images.source.src = frame.source;
  images.yoloe.src = frame.yoloe_image;
  images.sam.src = samView === 'overlay' ? frame.sam_image : frame.instance_map;
  elements.position.textContent = `${framePosition + 1} / ${scene.frame_count}`;
  elements.slider.value = framePosition;
  elements.frame.value = String(framePosition);
  elements.previous.disabled = framePosition === 0;
  elements.next.disabled = framePosition === scene.frame_count - 1;
  elements.range.textContent = `原始帧 ${frame.frame_index} · 当前场景范围 ${scene.first_frame}–${scene.last_frame}`;

  elements.badges.replaceChildren();
  const retained = frame.instances.filter(instance => instance.confidence >= confidenceThreshold);
  addBadge(`${retained.length} / ${frame.instance_count} 通过阈值`, true);
  addBadge(`${new Set(retained.map(instance => instance.class_name)).size} 类标签`);
  addBadge(`YOLOE ${formatSeconds(frame.yoloe_seconds)}`);
  addBadge(`SAM3.1 ${formatSeconds(frame.sam_seconds)}`);
  addBadge(`总计 ${formatSeconds(frame.wall_seconds)}`);

  renderNearbyFrames();
  setQuery();
}

function setFrame(position) {
  const scene = currentScene();
  framePosition = Math.max(0, Math.min(Number(position), scene.frame_count - 1));
  renderFrame();
}

function setScene(position, requestedFrame = null) {
  scenePosition = Math.max(0, Math.min(Number(position), data.scenes.length - 1));
  const scene = currentScene();
  elements.scene.value = String(scenePosition);
  elements.frame.replaceChildren();
  scene.frames.forEach((frame, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `frame ${String(frame.frame_index).padStart(6, '0')}`;
    elements.frame.appendChild(option);
  });
  elements.slider.max = String(scene.frame_count - 1);
  elements.range.textContent = `原始帧范围 ${scene.first_frame}–${scene.last_frame}`;
  const exactPosition = requestedFrame === null
    ? 0
    : scene.frames.findIndex(frame => frame.frame_index === requestedFrame);
  setFrame(exactPosition >= 0 ? exactPosition : 0);
  renderEpisodeSummary();
}

function setSamView(view) {
  samView = view;
  elements.overlay.classList.toggle('active', view === 'overlay');
  elements.idMap.classList.toggle('active', view === 'id-map');
  elements.samViewLabel.textContent = view === 'overlay'
    ? '逐实例 mask 叠加结果'
    : 'uint16 实例 ID 图';
  renderFrame();
}

function openLightbox(key) {
  const frame = currentFrame();
  const labels = {
    source: 'ScanNet RGB 输入',
    yoloe: 'YOLOE-26 检测框',
    sam: samView === 'overlay' ? 'SAM3.1 mask 叠加' : 'SAM3.1 实例 ID 图',
  };
  document.querySelector('#lightbox-image').src = images[key].src;
  document.querySelector('#lightbox-caption').textContent = `${currentScene().scene_id} · frame ${frame.frame_index} · ${labels[key]}`;
  document.querySelector('#lightbox').showModal();
}

function initialize() {
  if (!data?.scenes?.length) throw new Error('离线结果索引为空，请重新生成 data.js');
  document.querySelector('#scene-count').textContent = data.scene_count.toLocaleString();
  document.querySelector('#frame-count').textContent = data.frame_count.toLocaleString();
  document.querySelector('#instance-count').textContent = data.instance_count.toLocaleString();
  data.scenes.forEach((scene, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${scene.scene_id} · ${scene.frame_count} 帧`;
    elements.scene.appendChild(option);
  });

  const query = new URLSearchParams(window.location.search);
  const requestedScene = query.get('scene');
  const requestedFrame = Number.parseInt(query.get('frame') || '', 10);
  const requestedConfidence = Number.parseFloat(query.get('confidence') || '0.50');
  confidenceThreshold = Number.isNaN(requestedConfidence)
    ? 0.5
    : Math.min(0.95, Math.max(0.25, requestedConfidence));
  elements.confidence.value = confidenceThreshold.toFixed(2);
  elements.confidenceValue.value = confidenceThreshold.toFixed(2);
  const requestedScenePosition = data.scenes.findIndex(scene => scene.scene_id === requestedScene);
  setScene(requestedScenePosition >= 0 ? requestedScenePosition : 0, Number.isNaN(requestedFrame) ? null : requestedFrame);
}

elements.scene.addEventListener('change', event => setScene(event.target.value));
elements.frame.addEventListener('change', event => setFrame(event.target.value));
elements.slider.addEventListener('input', event => setFrame(event.target.value));
elements.previous.addEventListener('click', () => setFrame(framePosition - 1));
elements.next.addEventListener('click', () => setFrame(framePosition + 1));
elements.overlay.addEventListener('click', () => setSamView('overlay'));
elements.idMap.addEventListener('click', () => setSamView('id-map'));
elements.confidence.addEventListener('input', event => {
  confidenceThreshold = Number(event.target.value);
  elements.confidenceValue.value = confidenceThreshold.toFixed(2);
  renderFrame();
  renderEpisodeSummary();
});
elements.minimumFrames.addEventListener('change', () => renderEpisodeSummary());
document.querySelectorAll('.image-button').forEach(button => {
  button.addEventListener('click', () => openLightbox(button.dataset.image));
});
document.querySelector('#close-lightbox').addEventListener('click', () => document.querySelector('#lightbox').close());
document.querySelector('#lightbox').addEventListener('click', event => {
  if (event.target.id === 'lightbox') event.currentTarget.close();
});
window.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') setFrame(framePosition - 1);
  if (event.key === 'ArrowRight') setFrame(framePosition + 1);
  if (event.key === 'Escape') document.querySelector('#lightbox').close();
});

initialize();
