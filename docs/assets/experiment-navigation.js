(function(){
  // `path` is each experiment's directory relative to the repository root. The
  // pre-refactor tree's runtime was deleted on 2026-10-19, so every entry below
  // now lives under InstanceRecognition/site/experiments.
  const experiments=[
    {id:'harness-v3-sol-20frames-20261014-1313',label:'V3 · 前 20 帧 / 4 窗口 · 剪枝 mask · GPT-6 Sol · 2026-10-14',path:'InstanceRecognition/site/experiments'},
    {id:'harness-v4-mesh-sol-20frames-20261014-1558',label:'V4 · 前 20 帧 / 4 窗口 · mesh 首表面 · GPT-6 Sol · 2026-10-14',path:'InstanceRecognition/site/experiments'},
    {id:'harness-v5-amb3r-150frames-20261016-1550',label:'V5 · 前 150 帧 / 30 窗口 · AMB3R 逐帧深度 · GPT-6 Sol · 2026-10-16',path:'InstanceRecognition/site/experiments'}
  ];
  const aliases={};
  const path=decodeURIComponent(location.pathname).replace(/\\/g,'/');
  // Records and result pages are siblings inside the experiment directory, so one
  // base covers both page types. It is derived from the PAGE url, not the script url.
  const recordMatch=path.match(/\/(harness-[^/]+)\/records\.html$/);
  const resultMatch=path.match(/\/(harness-[^/]+)\/index\.html$/);
  const isRecords=!!recordMatch;
  let id=(recordMatch||resultMatch||[])[1];
  id=aliases[id]||id;
  if(!id) return;
  // Repo root, taken from this script's own url: it is loaded from docs/assets/ by
  // every page in both trees, so two levels up is the root regardless of the page.
  const self=document.currentScript&&document.currentScript.src;
  const root=self?new URL('../../',self).href:null;
  // Fall back to the page-derived base only for an entry registered at the same
  // depth as the current page -- used by experiments added on the fly below.
  const pageBase=new URL('../',location.href).href;
  const baseFor=e=>(e.path&&root)?new URL(e.path+'/',root).href:pageBase;
  // A future named experiment gets the same paired links without hard-coded
  // filenames. It has no registered path, so it assumes the current page's base.
  if(!experiments.some(e=>e.id===id)) experiments.push({id,label:id});
  const current=experiments.find(e=>e.id===id);
  const linkFor=(e,type)=>new URL(e.id+(type==='records'?'/records.html':'/index.html'),baseFor(e)).href;
  const nav=document.createElement('nav');nav.className='experimentNav';nav.setAttribute('aria-label','实验版本切换');
  const label=document.createElement('label');label.textContent='选择实验';
  const select=document.createElement('select');select.setAttribute('aria-label','选择实验版本');
  experiments.forEach(e=>{const o=document.createElement('option');o.value=e.id;o.textContent=e.label;o.selected=e.id===current.id;select.appendChild(o)});
  select.addEventListener('change',()=>{const e=experiments.find(x=>x.id===select.value);location.href=linkFor(e,isRecords?'records':'result');});
  const links=document.createElement('div');links.className='experimentNavLinks';
  const resultLink=document.createElement('a');resultLink.className='experimentNavLink';resultLink.textContent='场景理解结果';resultLink.href=linkFor(current,'result');
  const recordsLink=document.createElement('a');recordsLink.className='experimentNavLink';recordsLink.textContent='逐窗口交互记录';recordsLink.href=linkFor(current,'records');
  (isRecords?recordsLink:resultLink).setAttribute('aria-current','page');
  const hint=document.createElement('span');hint.className='experimentNavHint';hint.textContent='切换实验时保持当前页面类型';
  links.append(resultLink,recordsLink);nav.append(label,select,links,hint);
  const anchor=document.querySelector('header');
  if(anchor) anchor.appendChild(nav); else document.body.prepend(nav);
})();
