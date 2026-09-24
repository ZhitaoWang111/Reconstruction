"use strict";
const report=window.METHOD_COMPARISON;
const frames=report.frames;
const bySequence=new Map(frames.map(f=>[f.sequence_index,f]));
const left=document.getElementById('baseline'), right=document.getElementById('gpt');
const viewers=[left,right], ready=new Set(), loaded=new Map();
let sequence=frames[0].sequence_index;
const $=id=>document.getElementById(id);
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function send(viewer,data){viewer.contentWindow.postMessage(data,'*')}
function render(){
 const f=bySequence.get(sequence);
 $('sequence').value=String(sequence);$('slider').value=String(sequence);
 $('previous').disabled=sequence===frames[0].sequence_index;$('next').disabled=sequence===frames.at(-1).sequence_index;
 $('frameStatus').textContent=`抽样序号 ${sequence} / 源帧 ${f.source_frame_id} · ${viewers.every(v=>loaded.get(v)===sequence)?'两侧已同步':'正在同步两侧…'}`;
 $('baselineStats').textContent=`${f.baseline_masks} 个 mask · ${f.baseline_classes.length} 个原始类别`;
 $('gptStats').textContent=`${f.gpt_categories} 个类别提示 → ${f.gpt_raw_masks} 个原始 mask / ${f.gpt_masks} 个展示 mask · GPT ${f.gpt_seconds.toFixed(2)}s · SAM ${f.sam_seconds.toFixed(2)}s`;
 $('prompts').innerHTML=f.prompts.map(p=>`<tr><td><strong>${esc(p.category)}</strong><br><code>${esc(p.sam_prompt)}</code></td><td><span class="pill ${p.certainty==='uncertain'?'uncertain':''}">${p.certainty==='clear'?'明确':'不确定'}</span></td><td>${esc(({whole:'整体大致可见',partial:'局部可见',unclear:'范围不明'})[p.visible_scope])}</td><td>${p.raw_candidate_ids.length} / ${p.selected_count}${!p.raw_candidate_ids.length?' · 未分出 mask':''}</td><td>${esc(p.evidence)}${p.ambiguity?'<br>歧义：'+esc(p.ambiguity):''}</td></tr>`).join('');
 $('unresolved').textContent=f.unresolved_regions.length?'未识别区域：'+f.unresolved_regions.map(x=>x.location+' — '+x.reason).join('；'):'GPT 未报告无法识别的区域。';
 $('downloads').innerHTML=`<a href="${esc(f.gpt_labels)}">GPT 原始判断 JSON</a> · <a href="${esc(f.gpt_predictions)}">SAM 输出与原始 mask 路径</a> · <a href="${esc(f.baseline_predictions)}">基线输出 JSON</a>`;
}
function choose(value){if(!bySequence.has(value))return;sequence=value;render();for(const v of viewers)if(ready.has(v))send(v,{type:'mask-comparison:frame',sequence})}
frames.forEach(f=>{const option=document.createElement('option');option.value=f.sequence_index;option.textContent=`${f.sequence_index} / 源帧 ${f.source_frame_id}`;$('sequence').append(option)});
$('slider').min=frames[0].sequence_index;$('slider').max=frames.at(-1).sequence_index;
$('modelName').textContent=report.model.replace(/^openai\//,'');
$('sequence').onchange=e=>choose(Number(e.target.value));$('slider').oninput=e=>choose(Number(e.target.value));
$('previous').onclick=()=>choose(sequence-1);$('next').onclick=()=>choose(sequence+1);
$('rgb').onclick=()=>viewers.forEach(v=>send(v,{type:'mask-comparison:mode',mode:'rgb'}));
$('masks').onclick=()=>viewers.forEach(v=>send(v,{type:'mask-comparison:mode',mode:'all'}));
$('reset').onclick=()=>viewers.forEach(v=>send(v,{type:'mask-comparison:reset'}));
window.addEventListener('message',event=>{
 const v=viewers.find(x=>x.contentWindow===event.source);if(!v)return;
 if(event.data?.type==='mask-comparison:height' && Number.isFinite(event.data.height) && event.data.height>200 && event.data.height<4000){v.style.height=(event.data.height+2)+'px';}
 if(event.data?.type==='mask-comparison:ready'){ready.add(v);send(v,{type:'mask-comparison:frame',sequence});}
 if(event.data?.type==='mask-comparison:loaded'&&bySequence.has(event.data.sequence)){
   loaded.set(v,event.data.sequence);render();
 }
});
render();
