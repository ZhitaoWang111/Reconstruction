// null keeps the existing all/single-mask mode; an empty Set shows no objects.
let selectedObjectIds=null;
let objectListView="windows";
function objectIsChecked(id){return selectedObjectIds!==null?selectedObjectIds.has(id):selectedObjectId===null||selectedObjectId===id;}
/* Group real committed changes by window; never synthesize model decisions. */
drawObjectList=function(){
 const list=$('objectList');list.replaceChildren();list.classList.add('windowGroups');
 const active=new Map(stateCloud.snapshot.objects.map(o=>[o.id,o]));
 const currentWindow=RESULT_FRAMES[currentFrameIndex].window_index;
 const groups=objectListView==='all'?[{window_index:currentWindow,objects:[...active.values()].map(o=>({object_id:o.id,object_label:o.object_label,changes:[]}))}]:RESULT_CHANGES;
 for(const group of groups){
  if(group.window_index>RESULT_FRAMES[currentFrameIndex].window_index)break;
  const section=document.createElement('section');section.className='objectWindow';section.dataset.windowIndex=group.window_index;
  const title=document.createElement('h3');title.textContent=objectListView==='all'?`当前帧对应的最新状态 · ${active.size} 个活跃物体`:`窗口 ${group.window_index+1} · 抽样帧 ${group.first}–${group.cutoff}`;section.appendChild(title);
  const grid=document.createElement('div');grid.className='objectWindowGrid';section.appendChild(grid);
  for(const item of [...group.objects].sort((a,b)=>Number(a.object_id)-Number(b.object_id))){
   const o=active.get(item.object_id),b=document.createElement('button');b.className='objectItem'+(objectIsChecked(item.object_id)?' active':'');b.dataset.objectId=item.object_id;
   b.style.borderLeft=`4px solid ${o?'rgb('+o.color.join(',')+')':'#9aa4b4'}`;
   b.innerHTML=`<span>Object ${String(item.object_id).padStart(3,'0')} · ${escapeHtml(item.object_label)}</span><small class="changeTag">${escapeHtml(item.changes.join('；'))}</small><small>${o?`当前：${escapeHtml(o.object_label)} · ${o.frames.length} 帧 / ${o.voxel_count.toLocaleString()} 点`:'已合并，不再单独显示点云'}</small>`;
   b.disabled=!o;b.setAttribute("aria-pressed",String(!!o&&objectIsChecked(item.object_id)));
   b.onclick=()=>{
    if(selectedObjectIds===null)selectedObjectIds=new Set(selectedObjectId===null?stateCloud.snapshot.objects.map(x=>x.id):selectedObjectId>0?[selectedObjectId]:[]);
    if(selectedObjectIds.has(o.id))selectedObjectIds.delete(o.id);else selectedObjectIds.add(o.id);
    selectedObjectId=null;selectedInstanceIndices.clear();
    for(const a of stateCloud.snapshot.assignments)if(selectedObjectIds.has(a.object_id))selectedInstanceIndices.add(manifest.frames[currentFrameIndex].instances.findIndex(x=>x.instance_id===a.instance_id));
    imageMode='selected';syncModeButtons();drawObjectList();draw2d(currentImage);updateInstanceSelection();updateCloudBuffers();renderCloud();
   };grid.appendChild(b);
  }
  if(!group.objects.length){const p=document.createElement('p');p.textContent='本窗口无物体变更';grid.appendChild(p);}
  list.appendChild(section);
 }
 $('objectStats').textContent=`截至窗口 ${RESULT_FRAMES[currentFrameIndex].window_index+1} · ${active.size} 个活跃物体`;
 $('objectsByWindow').setAttribute('aria-pressed',String(objectListView==='windows'));
 $('objectsAll').setAttribute('aria-pressed',String(objectListView==='all'));
 $('objectsByWindow').classList.toggle('active',objectListView==='windows');
 $('objectsAll').classList.toggle('active',objectListView==='all');
 $('allObjects').classList.toggle('active',selectedObjectIds===null&&selectedObjectId===null);
};

// Read-only legend maps frame-local mask IDs to upstream category labels.
drawInstances=function(){
 const list=$('instanceList');list.replaceChildren();
 const frame=manifest.frames[currentFrameIndex],row=RESULT_FRAMES[currentFrameIndex];
 for(const [i,item] of frame.instances.entries()){
  const entry=document.createElement('div');entry.className='instanceItem maskLegendItem';
  const color=row.input_colors[String(item.instance_id)];
  const maskId=row.input_mask_ids?.[String(item.instance_id)]||`M${String(i+1).padStart(2,'0')}`;
  entry.innerHTML=`<span class="swatch" style="background:rgb(${color.join(',')})"></span><span class="name"><strong>${escapeHtml(maskId)}</strong> · ${escapeHtml(item.class_name)}</span>`;
  list.appendChild(entry);
 }
 updateInstanceSelection();
};
updateInstanceSelection=function(){
 $('instanceSummary').textContent=`${manifest.frames[currentFrameIndex].instances.length} 个 mask`;
};
const updateMultiSelection=updateCloudBuffers;
updateCloudBuffers=function(){updateMultiSelection();if(selectedObjectIds!==null){const ids=stateCloud.snapshot.objects.filter(o=>selectedObjectIds.has(o.id)).map(o=>o.id);$('selectionInfo').textContent=ids.length?`已选中 ${ids.length} 个物体：${ids.map(id=>'Object '+String(id).padStart(3,'0')).join('、')}`:'未选中物体';}};
$('allObjects').addEventListener('click',()=>{selectedObjectIds=null;},true);

$('objectsByWindow').onclick=()=>{objectListView='windows';drawObjectList();};
$('objectsAll').onclick=()=>{objectListView='all';drawObjectList();};

$('selectAllObjects').onclick=()=>$('allObjects').click();
$('deselectAllObjects').onclick=()=>{
 selectedObjectIds=new Set();selectedObjectId=null;selectedInstanceIndices.clear();
 imageMode='selected';syncModeButtons();drawObjectList();draw2d(currentImage);
 updateInstanceSelection();updateCloudBuffers();renderCloud();
};
