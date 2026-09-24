#!/usr/bin/env python3
"""Real offline-browser checks for the two-method comparison and shared frame controls."""
import argparse, json, subprocess, tempfile, time, urllib.request
from pathlib import Path
from check_mask_pointcloud_viewer import Browser

class FrameBrowser(Browser):
    def __init__(self,url):
        super().__init__(url); self.contexts={}
    def call(self,method,params=None):
        self.next_id+=1; rid=self.next_id
        self.socket.send(json.dumps({'id':rid,'method':method,'params':params or {}}))
        while True:
            message=json.loads(self.socket.recv())
            if message.get('method')=='Runtime.executionContextCreated':
                ctx=message['params']['context']; aux=ctx.get('auxData',{})
                if aux.get('isDefault'): self.contexts[aux.get('frameId')]=ctx['id']
            if message.get('method')=='Runtime.exceptionThrown': self.exceptions.append(message['params'])
            if message.get('id')==rid:
                if 'error' in message: raise RuntimeError(message['error'])
                return message.get('result',{})
    def frame_js(self,frame_id,expression):
        result=self.call('Runtime.evaluate',{'expression':expression,'contextId':self.contexts[frame_id],
            'returnByValue':True,'awaitPromise':True})
        if 'exceptionDetails' in result: raise RuntimeError(result['exceptionDetails'])
        return result['result'].get('value')
    def frame_wait(self,frame_id,expression,seconds=40):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if self.frame_js(frame_id,expression): return
            time.sleep(.1)
        raise TimeoutError(expression)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--page',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    profile=Path(tempfile.mkdtemp(prefix='method-comparison-chrome-'));log=(args.output/'browser.log').open('wb')
    p=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-dev-shm-usage',
        '--disable-background-networking','--disable-extensions','--no-first-run','--enable-unsafe-swiftshader',
        '--use-angle=swiftshader','--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
    b=None; results={'page':str(args.page.resolve()),'frames':[]}
    try:
        deadline=time.monotonic()+30
        while not (profile/'DevToolsActivePort').exists() and time.monotonic()<deadline:
            assert p.poll() is None,'Chrome exited';time.sleep(.1)
        port=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as r:
            target=next(t for t in json.load(r) if t['type']=='page')
        b=FrameBrowser(target['webSocketDebuggerUrl']);b.call('Runtime.enable');b.call('Page.enable')
        b.call('Emulation.setDeviceMetricsOverride',{'width':1900,'height':1400,'deviceScaleFactor':1,'mobile':False})
        b.call('Page.navigate',{'url':args.page.resolve().as_uri()})
        b.wait("document.getElementById('frameStatus')?.textContent.includes('两侧已同步')",seconds=60)
        tree=b.call('Page.getFrameTree')['frameTree']
        child={('gpt' if '/gpt/' in x['frame']['url'] else 'baseline'):x['frame']['id'] for x in tree['childFrames']}
        assert set(child)=={'gpt','baseline'}
        assert b.js("document.querySelectorAll('#sequence option').length")==61
        b.screenshot(args.output/'desktop_0080.png')
        expected=json.loads((args.page.parent/'comparison.json').read_text())
        for seq in (80,95,110,111,112,113,125,140):
            b.js(f"document.getElementById('sequence').value='{seq}';document.getElementById('sequence').dispatchEvent(new Event('change'))")
            b.wait(f"document.getElementById('frameStatus').textContent.startsWith('抽样序号 {seq} /') && document.getElementById('frameStatus').textContent.includes('两侧已同步')")
            entry=next(f for f in expected['frames'] if f['sequence_index']==seq); check={'sequence':seq}
            for method,fid in child.items():
                b.frame_wait(fid,f"manifest.frames[currentFrameIndex].sequence_index==={seq} && currentCloud.sourceFrameId==={entry['source_frame_id']}")
                stats=b.frame_js(fid,"({source:currentCloud.sourceFrameId,masks:manifest.frames[currentFrameIndex].instances.length,points:currentCloud.positions.length/3,drawn:glState.drawCount,background:glState.bgCount,error:glState.gl.getError(),imageReady:currentImage.complete&&currentImage.naturalWidth>0})")
                assert stats['masks']==entry['baseline_masks' if method=='baseline' else 'gpt_masks'],stats
                assert stats['points']==stats['drawn'] and stats['error']==0 and stats['background']>0 and stats['imageReady'],stats
                check[method]=stats
            results['frames'].append(check)
            if seq in (111,112,113):
                b.screenshot(args.output/f'desktop_{seq:04d}.png')
            print('BROWSER sequence='+str(seq),flush=True)
        # Exercise instance selection and both sides' shared mode controls.
        b.js("document.getElementById('sequence').value='112';document.getElementById('sequence').dispatchEvent(new Event('change'))")
        b.wait("document.getElementById('frameStatus').textContent.startsWith('抽样序号 112 /') && document.getElementById('frameStatus').textContent.includes('两侧已同步')")
        for method,fid in child.items():
            b.frame_js(fid,"document.querySelector('.instanceItem').click()")
            stats=b.frame_js(fid,"({selected:[...selectedInstanceIndices],mode:imageMode,drawn:glState.drawCount,expected:currentCloud.maskBits.reduce((n,b)=>n+((b&1)?1:0),0),labels:document.querySelectorAll('.maskLabel').length})")
            assert stats['selected']==[0] and stats['mode']=='selected' and stats['drawn']==stats['expected'],stats
            results[method+'_selection']=stats
        b.js("document.getElementById('rgb').click()")
        for fid in child.values(): b.frame_wait(fid,"imageMode==='rgb' && document.querySelectorAll('.maskLabel').length===0")
        b.js("document.getElementById('masks').click()")
        for fid in child.values(): b.frame_wait(fid,"imageMode==='all' && selectedInstanceIndices.size===0")
        b.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':1000,'deviceScaleFactor':1,'mobile':False})
        time.sleep(.2)
        assert b.js('document.documentElement.scrollWidth<=innerWidth'),'Mobile overflow'
        b.screenshot(args.output/'mobile_top.png')
        b.js("document.getElementById('gpt').scrollIntoView({behavior:'instant'})")
        time.sleep(.2);b.screenshot(args.output/'mobile_gpt.png')
        assert not b.exceptions,b.exceptions
        results['exceptions']=b.exceptions;results['mobile_no_overflow']=True
        (args.output/'checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print('PASS: offline loading, synchronized frames, masks, 3D, selection, shared modes, mobile')
    finally:
        if b:b.socket.close()
        p.terminate()
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        log.close()

if __name__=='__main__':main()
