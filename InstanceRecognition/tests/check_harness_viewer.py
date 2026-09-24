"""Offline browser and causal-state QA for the Harness experiment."""
import json,subprocess,tempfile,time,urllib.request
from pathlib import Path
import numpy as np
from check_mask_pointcloud_viewer import Browser
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'InstanceRecognition/runtime/experiments/harness_first30_v1'
def main():
 qa=OUT/'qa';qa.mkdir(exist_ok=True);checks={}
 total=json.loads((OUT/'checkpoint.json').read_text())['version']
 observations=json.loads((OUT/'observations.json').read_text())
 for v in range(1,total+1):
  s=json.loads((OUT/f'versions/{v:04d}/state.json').read_text())
  assert set(s['assignments'])=={k for k,o in observations.items() if o['seq']<=s['cutoff']}
  with np.load(OUT/f'states/{v-1:04d}.npz') as z:
   assert len(z['points_world'])==len(z['object_ids'])
   assert np.isfinite(z['points_world']).all()
   assert set(z['object_ids'])=={int(o[1:]) for o in s['objects']}
 checks['causal_states']=total
 profile=Path(tempfile.mkdtemp(prefix='harness-chrome-'));log=(qa/'browser.log').open('wb')
 proc=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--disable-extensions','--no-first-run','--enable-unsafe-swiftshader','--use-angle=swiftshader','--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
 b=None
 try:
  deadline=time.monotonic()+30
  while not (profile/'DevToolsActivePort').exists() and time.monotonic()<deadline:time.sleep(.1)
  port=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as f:targets=json.load(f)
  b=Browser(next(t for t in targets if t['type']=='page')['webSocketDebuggerUrl'])
  b.call('Runtime.enable');b.call('Page.enable');b.call('Emulation.setDeviceMetricsOverride',dict(width=1700,height=1200,deviceScaleFactor=1,mobile=False))
  b.call('Page.navigate',dict(url=(OUT/'index.html').as_uri()))
  b.wait("typeof stateCloud!=='undefined' && stateCloud && $('status').textContent.startsWith('已加载')")
  for i in range(total):
   b.js(f'loadFrame({i})')
   assert b.js('glState.drawCount===stateCloud.ids.length && glState.gl.getError()===0')
   assert b.js('stateCloud.snapshot.objects.every(o=>o.frames.every(f=>f<=stateCloud.snapshot.cutoff))')
  checks['rendered_windows']=total;b.screenshot(qa/'last_window.png')
  b.js("document.querySelector('[data-object-id]').click()")
  assert b.js('glState.drawCount===stateCloud.snapshot.objects.find(o=>o.id===selectedObjectId).voxel_count')
  b.screenshot(qa/'selected_object.png');checks['object_selection']=True
  b.js("$('allObjects').click();loadFrame(0)")
  assert b.js('currentFrameIndex===0 && glState.drawCount===stateCloud.ids.length')
  assert b.js('document.documentElement.scrollWidth<=innerWidth')
  b.call('Emulation.setDeviceMetricsOverride',dict(width=440,height=900,deviceScaleFactor=1,mobile=True));time.sleep(.3)
  assert b.js('document.documentElement.scrollWidth<=innerWidth');b.screenshot(qa/'mobile.png')
  b.call('Page.navigate',dict(url=(OUT/'audit.html').as_uri()));b.wait("document.querySelectorAll('article').length>1")
  assert b.js(f"document.querySelectorAll('option').length==={total}")
  b.js(f"document.getElementById('window').value={total-1};document.getElementById('window').dispatchEvent(new Event('change'))")
  b.wait("Array.from(document.images).filter(i=>i.getBoundingClientRect().top<innerHeight).every(i=>i.complete&&i.naturalWidth>0)")
  assert b.js('document.documentElement.scrollWidth<=innerWidth');b.screenshot(qa/'audit_mobile.png')
  assert not b.exceptions,b.exceptions
  checks.update(responsive=True,audit=True,runtime_exceptions=b.exceptions,passed=True)
 finally:
  if b:b.socket.close()
  proc.terminate()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close();(qa/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
