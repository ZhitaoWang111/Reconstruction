"""Build the isolated frame-6 prompt/grouping experiment using the existing 3D viewer."""
import copy,json,os
from pathlib import Path
from src.visualization.build_method_comparison import viewer_bundle
from src.visualization.publish_gpt_viewer import rebase_manifest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runtime/experiments/frame0006_pairing'
API=ROOT.parent.parent/'API/scannet_gpt_sam/runs'

def main():
 # This historical experiment must keep the original singular-prompt results,
 # even after the primary viewer advances to a newer prompt revision.
 source=ROOT/'runtime/mask_pointclouds/scene0000_00/gpt_terra_gt_raw'
 manifest=json.loads((source/'manifest.json').read_text());frame=manifest['frames'][6]
 assert frame['sequence_index']==6 and frame['source_frame_id']==67
 pairing=json.loads((API/'scene0000_00_0006_mask_pairing/pairing.json').read_text())
 groups=pairing['result']['groups'];assert sorted(x for g in groups for x in g['member_ids'])==[2,3,4,5]
 for name in ('original_viewer','group_viewer'):
  dest=OUT/name;m=copy.deepcopy(manifest);m.update(frames=[copy.deepcopy(frame)],frame_count=1,mask_count=len(frame['instances']))
  m=rebase_manifest(m,source,dest)
  if name=='group_viewer':
   m['semantic_groups']=groups;m['grouping_policy']='Selection groups only; independent masks, NPZ memberships and geometry unchanged'
   for index,g in enumerate(groups):
    for instance in m['frames'][0]['instances']:
     if instance['instance_id'] in g['member_ids']:instance['color']=[[86,213,180],[211,159,255]][index%2]
  viewer_bundle(dest,m,'gpt')
  if name=='group_viewer':
   html=(dest/'index.html').read_text().replace('本帧原始实例','本帧语义分组').replace('<script src="bridge.js"></script>','<script src="bridge.js"></script>\n<script src="groups.js"></script>')
   html=html.replace('点击类别显示对应 mask，可多选；再次点击取消。','点击“一双鞋”同时选择它的两张原始 mask；可多选。')
   (dest/'index.html').write_text(html)
   (dest/'groups.js').write_text(GROUP_JS)
 prompt=json.loads((API/'scene0000_00_0006_pair_prompt_v2/labels/0006.json').read_text())
 sam=json.loads((OUT/'prompt_v2_results/frame_000067/sam31/predictions.json').read_text())
 record={'sequence_index':6,'source_frame_id':67,'original_shoe_masks':4,'prompt_only_shoe_masks':sum(x['class_name']=='shoes' for x in sam['instances']),
   'semantic_pairs':groups,'unpaired':pairing['result']['unpaired'],'grouping_inputs':'original RGB plus numbered crop of the original four SAM masks; no target number of pairs',
   'raw_masks_merged':False,'production_changed':False,'prompt_labels':prompt['labels'],'pairing_response':str(API/'scene0000_00_0006_mask_pairing/pairing.json')}
 direct_path=OUT/'direct_pair_results/frame_000067/sam31/predictions.json'
 direct=None
 if direct_path.exists():
  direct=json.loads(direct_path.read_text())
  record['direct_pair_prompt']={'mask_count':sum(x['class_name']=='pair of shoes' for x in direct['instances']),
   'intervention':direct['intervention'],'validation':json.loads((OUT/'direct_pair_validation.json').read_text())}
  dest=OUT/'direct_pair_viewer'
  direct_html=(dest/'index.html').read_text().replace('GPT-5.6 Terra + SAM3.1 / AMB3R GT-corrected','SAM3.1 · 直接指定 pair of shoes / AMB3R GT-corrected')
  (dest/'index.html').write_text(direct_html)
 (OUT/'experiment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 links=' · '.join(f'<a href="{os.path.relpath(p,OUT)}">{title}</a>' for title,p in [('改写后的英文 prompt',API/'scene0000_00_0006_pair_prompt_v2/prompt_en.txt'),('配对英文 prompt',API/'scene0000_00_0006_mask_pairing/prompt_en.txt'),('GPT 配对 JSON',API/'scene0000_00_0006_mask_pairing/pairing.json')])
 html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 6 帧 · 鞋的实例与配对</title>
<style>body{margin:0;background:#0b1119;color:#eef4ff;font:16px/1.6 system-ui}main{max-width:1600px;margin:auto;padding:24px}h1{font-size:28px;margin:0}p{color:#aebcd0}button,a{color:#69d5f6}button{padding:10px 16px;background:#172331;border:1px solid #435268;border-radius:8px;cursor:pointer}button.active{background:#31515f;color:white}nav{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}iframe{width:100%;height:1250px;border:1px solid #384557;border-radius:10px;background:#0b1119}figure{margin:20px 0}img{max-width:100%;border-radius:10px}details{margin:20px 0}code{color:#bde2ff}@media(max-width:600px){main{padding:12px}iframe{height:1700px}}</style>
<main><h1>第 6 帧：四只鞋，怎样表示为两双？</h1><p>抽样序号 6 / 原始帧 67 · GPT-5.6 Terra + SAM3.1 · 单帧实验</p>
<p>原方案得到 4 张鞋 mask；仅放宽单复数要求后，本次得到 3 张鞋 mask。编号 mask 配对得到两组：<strong>[2, 5]</strong> 与 <strong>[3, 4]</strong>。分组只改变选择方式，保留四张原始 mask 和原有 3D 落点。</p>
<nav><button data-src="group_viewer/index.html" class="active">按双分组 · 2 双</button><button data-src="original_viewer/index.html">原方案 · 4 张鞋 mask</button><button data-src="prompt_viewer/index.html">仅改 prompt · 3 张鞋 mask</button></nav>
<p id="hint">点击“一双鞋”会同时选中对应的两张 mask，可在 2D 和 3D 中检查。</p><iframe id="viewer" title="单帧实例与 3D 检查" src="group_viewer/index.html"></iframe>
<details><summary>配对输入与实验记录</summary><p>GPT 同时看到原始 RGB 和下方带编号的局部图；未告知应输出几双。它根据外观、鞋底和左右对应关系分组。本次为单帧验证，不能据此推断所有场景都能可靠配对。</p><figure><img src="numbered_candidates.jpg" alt="四张原始鞋 mask 的编号"><figcaption>编号与原始 mask ID 一致。</figcaption></figure>LINKS · <a href="experiment.json">实验记录</a></details>
<p>全场景正式页面与原始 prompt 未修改。</p></main>
<script>const frame=document.getElementById('viewer');for(const b of document.querySelectorAll('nav button'))b.onclick=()=>{for(const x of document.querySelectorAll('nav button'))x.classList.toggle('active',x===b);frame.src=b.dataset.src;document.getElementById('hint').textContent=b.dataset.src.startsWith('group_')?'点击“一双鞋”会同时选中对应的两张原始 mask。':'点击实例，检查该张 mask 的 2D 覆盖范围和 3D 落点。'};window.addEventListener('message',e=>{if(e.source===frame.contentWindow&&e.data?.type==='mask-comparison:height'&&e.data.height>200&&e.data.height<4000)frame.style.height=(e.data.height+4)+'px'});</script></html>'''.replace('LINKS',links)
 if direct:
  count=record['direct_pair_prompt']['mask_count']
  html=html.replace('<nav><button data-src="group_viewer/index.html" class="active">',f'<p><strong>最新实验：直接指定 <code>pair of shoes</code>，SAM 输出 {count} 张鞋类 mask。</strong>本帧分别覆盖左右两双；没有配对或合并后处理。复用原 GPT 类别，仅手动替换鞋类名称与 SAM 提示词，未新增 GPT 请求。</p><nav><button data-src="direct_pair_viewer/index.html" class="active">直接 pair of shoes · {count} 张 mask</button><button data-src="group_viewer/index.html">')
  html=html.replace('src="group_viewer/index.html"></iframe>','src="direct_pair_viewer/index.html"></iframe>')
  html=html.replace('<p id="hint">点击“一双鞋”会同时选中对应的两张 mask，可在 2D 和 3D 中检查。</p>','<p id="hint">点击 pair of shoes 实例，检查 SAM 直接生成的单张 mask 是否覆盖完整一双。</p>')
  html=html.replace('仅改 prompt · 3 张鞋 mask','放宽单复数 · 3 张鞋 mask')
  html=html.replace('<p>全场景正式页面与原始 prompt 未修改。</p>','<p><a href="direct_pair_validation.json">直接提示覆盖检查</a> · <a href="direct_pair_results/frame_000067/sam31/predictions.json">SAM 原始结果</a></p><p>本结果仅验证 SAM 在这一帧对该短语的响应，尚未验证 GPT 自动选择该短语及跨帧稳定性。全场景正式页面与原始 prompt 未修改。</p>')
 (OUT/'index.html').write_text(html)
 print(OUT/'index.html')

GROUP_JS='''"use strict";
const semanticGroups=window.MASK_PROJECTION_MANIFEST.semantic_groups;
const baseDrawGroups=drawInstances,baseUpdateGroups=updateInstanceSelection;
function groupIndices(g){return g.member_ids.map(id=>manifest.frames[currentFrameIndex].instances.findIndex(x=>x.instance_id===id));}
updateInstanceSelection=function(){baseUpdateGroups();for(const b of document.querySelectorAll('[data-group-index]')){const g=semanticGroups[Number(b.dataset.groupIndex)];const active=groupIndices(g).every(i=>selectedInstanceIndices.has(i));b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));}document.getElementById('instanceSummary').textContent=`${semanticGroups.length} 双鞋 · 保留 4 张原始鞋 mask`;};
drawInstances=function(){baseDrawGroups();const list=document.getElementById('instanceList');semanticGroups.forEach((g,n)=>{const indices=groupIndices(g);for(const i of indices)list.querySelector(`[data-instance-index="${i}"]`)?.remove();const b=document.createElement('button');b.type='button';b.className='instanceItem';b.dataset.groupIndex=String(n);const color=manifest.frames[currentFrameIndex].instances[indices[0]].color;b.innerHTML=`<span class="selectionMark">✓</span><span class="swatch" style="background:rgb(${color.join(',')})"></span><span class="name">一双鞋 ${n+1} · pair of shoes</span><span class="detail">原始 mask ${g.member_ids.join(' + ')}</span>`;b.title=g.evidence;b.onclick=()=>{const remove=indices.every(i=>selectedInstanceIndices.has(i));for(const i of indices){if(remove)selectedInstanceIndices.delete(i);else selectedInstanceIndices.add(i);}setImageMode('selected');};list.insertBefore(b,list.querySelector('[data-instance-index]'));});updateInstanceSelection();};
'''
if __name__=='__main__':main()
