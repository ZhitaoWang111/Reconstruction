"""Actual file:// checks for frame-6 pairing, prompt-only control, and raw-mask preservation."""
import json,subprocess,tempfile,time,urllib.request
from pathlib import Path
from check_method_comparison import FrameBrowser

root=Path(__file__).resolve().parents[1]/'runtime/experiments/frame0006_pairing';out=root/'browser';out.mkdir(exist_ok=True)
profile=Path(tempfile.mkdtemp(prefix='pairing-check-'));log=(out/'chrome.log').open('wb')
p=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--disable-extensions','--no-first-run','--enable-unsafe-swiftshader','--use-angle=swiftshader','--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
b=None
try:
 deadline=time.monotonic()+30
 while not (profile/'DevToolsActivePort').exists() and time.monotonic()<deadline:time.sleep(.1)
 port=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as response:target=next(x for x in json.load(response) if x['type']=='page')
 b=FrameBrowser(target['webSocketDebuggerUrl']);b.call('Runtime.enable');b.call('Page.enable');b.call('Emulation.setDeviceMetricsOverride',{'width':1800,'height':1400,'deviceScaleFactor':1,'mobile':False})
 b.call('Page.navigate',{'url':(root/'index.html').as_uri()})
 def child(name):
  deadline=time.monotonic()+45
  while time.monotonic()<deadline:
   tree=b.call('Page.getFrameTree')['frameTree']
   for frame in tree.get('childFrames',[]):
    fid=frame['frame']['id']
    if name in frame['frame']['url'] and fid in b.contexts:
     try:
      if b.frame_js(fid,"typeof currentCloud!=='undefined' && !!currentCloud && document.getElementById('status').textContent.startsWith('已加载')"):return fid
     except RuntimeError:pass
   time.sleep(.1)
  raise TimeoutError(name)
 fid=child('direct_pair_viewer');b.frame_wait(fid,"imageMode==='all'")
 assert b.frame_js(fid,"manifest.frames[0].instances.filter(x=>x.class_name==='pair of shoes').length")==2
 direct_checks=[]
 for n in (1,2):
  b.frame_js(fid,f"setImageMode('all');document.querySelector('[data-instance-index=\"{n}\"]').click()")
  info=b.frame_js(fid,"({selected:[...selectedInstanceIndices],drawn:glState.drawCount,expected:currentCloud.maskBits.reduce((n,b)=>n+((b&selectedMaskBits())?1:0),0),error:glState.gl.getError()})")
  assert info['selected']==[n] and info['drawn']==info['expected']>0 and info['error']==0,info
  direct_checks.append(info);b.screenshot(out/f'direct_pair_{n}.png')
 b.js("document.querySelector('button[data-src=\"group_viewer/index.html\"]').click()")
 fid=child('group_viewer');b.frame_wait(fid,"imageMode==='all'")
 assert b.frame_js(fid,"document.querySelectorAll('[data-group-index]').length")==2
 result={'pairs':[],'direct_pair_prompt':direct_checks}
 for n in (0,1):
  b.frame_js(fid,f"setImageMode('all');document.querySelector('[data-group-index=\"{n}\"]').click()")
  info=b.frame_js(fid,"({ids:[...selectedInstanceIndices].map(i=>manifest.frames[0].instances[i].instance_id).sort((a,b)=>a-b),drawn:glState.drawCount,expected:currentCloud.maskBits.reduce((n,b)=>n+((b&selectedMaskBits())?1:0),0),error:glState.gl.getError()})")
  assert info['ids']==([2,5] if n==0 else [3,4]) and info['drawn']==info['expected']>0 and info['error']==0,info
  result['pairs'].append(info);b.screenshot(out/f'pair_{n+1}.png')
 b.frame_js(fid,"document.querySelector('[data-group-index=\"1\"]').click()")
 assert b.frame_js(fid,'selectedInstanceIndices.size')==0
 for name,count in [('original_viewer',4),('prompt_viewer',3)]:
  b.js(f"document.querySelector('button[data-src=\"{name}/index.html\"]').click()")
  fid=child(name);b.frame_wait(fid,"imageMode==='all'")
  actual=b.frame_js(fid,"manifest.frames[0].instances.filter(x=>x.class_name==='shoe'||x.class_name==='shoes').length")
  assert actual==count,(name,actual);result[name]=actual
 b.js("document.querySelector('button[data-src=\"group_viewer/index.html\"]').click()")
 fid=child('group_viewer');b.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':1000,'deviceScaleFactor':1,'mobile':False});time.sleep(.3)
 assert b.js('document.documentElement.scrollWidth<=innerWidth')
 assert b.frame_js(fid,'document.documentElement.scrollWidth<=innerWidth')
 b.screenshot(out/'mobile.png');assert not b.exceptions,b.exceptions
 result.update(passed=True,exceptions=b.exceptions,mobile_no_overflow=True)
 (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:
 if b:b.socket.close()
 p.terminate()
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:p.kill();p.wait()
