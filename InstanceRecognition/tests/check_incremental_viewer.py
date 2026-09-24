"""Check causal snapshots and the real offline browser replay."""
import json,subprocess,tempfile,time,urllib.request
from pathlib import Path
import numpy as np
from check_mask_pointcloud_viewer import Browser

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'InstanceRecognition/runtime/experiments/incremental_objects_first30_v2'

def main():
    qa=OUT/'qa';qa.mkdir(exist_ok=True);checks={};seen=set();previous={}
    for seq in range(30):
        state=json.loads((OUT/f'states/{seq:04d}.json').read_text())
        with np.load(OUT/f'states/{seq:04d}.npz') as z:
            assert not np.any(z['support_frame_bits'] >> (seq+1))
            assert np.all(z['added_at']<=seq) and np.all(z['first_seen']<=seq)
            assert int((z['added_at']==seq).sum())==state['new_voxels']
            assert len(z['object_ids'])==state['voxel_count']
            for o in state['objects']:
                assert o['created_at']<=seq and max(o['frames'])<=seq
                if o['id'] in previous:assert set(previous[o['id']])<=set(o['observations'])
                previous[o['id']]=o['observations']
                sel=z['object_ids']==o['id']
                assert int(sel.sum())==o['voxel_count']
        seen.update(o['id'] for o in state['objects'])
    checks['causal_snapshots']=30;checks['final_objects']=len(seen)
    profile=Path(tempfile.mkdtemp(prefix='incremental-chrome-'))
    log=(qa/'browser.log').open('wb')
    proc=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-dev-shm-usage',
        '--disable-background-networking','--disable-extensions','--no-first-run',
        '--enable-unsafe-swiftshader','--use-angle=swiftshader','--remote-debugging-port=0',
        f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log)
    b=None
    try:
        deadline=time.monotonic()+30
        while not (profile/'DevToolsActivePort').exists() and time.monotonic()<deadline:time.sleep(.1)
        port=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as f:targets=json.load(f)
        b=Browser(next(t for t in targets if t['type']=='page')['webSocketDebuggerUrl'])
        b.call('Runtime.enable');b.call('Page.enable')
        b.call('Emulation.setDeviceMetricsOverride',dict(width=1700,height=1200,deviceScaleFactor=1,mobile=False))
        b.call('Page.navigate',dict(url=(OUT/'index.html').as_uri()))
        b.wait("typeof stateCloud!=='undefined' && stateCloud && $('status').textContent.startsWith('已加载')")
        for seq in range(30):
            b.js(f'loadFrame({seq})')
            assert b.js('glState.drawCount===stateCloud.ids.length && glState.gl.getError()===0')
        checks['browser_frames']=30
        b.screenshot(qa/'frame29_all.png')
        for name,expression in [('new','stateCloud.first[i]===currentFrameIndex'),
            ('supported','popcount32(stateCloud.support[i])>=3'),
            ('tentative','popcount32(stateCloud.support[i])<3'),('conflict','stateCloud.conflicts[i]')]:
            b.js(f"$('cloudFilter').value='{name}';$('cloudFilter').dispatchEvent(new Event('change'))")
            count=b.js(f'Array.from(stateCloud.ids).filter((_,i)=>{expression}).length')
            assert b.js('glState.drawCount')==count
            checks[name+'_points']=count
        b.screenshot(qa/'frame29_conflicts.png')
        b.js("$('cloudFilter').value='all';$('cloudFilter').dispatchEvent(new Event('change'));document.querySelector('[data-object-id=\"31\"]').click()")
        assert b.js('glState.drawCount===stateCloud.snapshot.objects.find(o=>o.id===31).voxel_count')
        assert b.js("$('modeSelectedMasks').classList.contains('active')")
        b.screenshot(qa/'frame29_chair31.png')
        b.js("$('allObjects').click();loadFrame(6)")
        b.wait("currentFrameIndex===6 && $('status').textContent.startsWith('已加载')")
        assert b.js('stateCloud.snapshot.objects.every(o=>o.created_at<=6 && o.frames.every(f=>f<=6))')
        b.screenshot(qa/'frame06_pairs.png')
        b.js("document.querySelectorAll('#instanceList button')[stateCloud.snapshot.assignments.findIndex(a=>a.object_id===null)].click()")
        assert b.js('selectedObjectId===0 && glState.drawCount>0')
        checks['pending_raw_selection']=True
        b.js("$('allObjects').click();loadFrame(29)")
        b.wait("currentFrameIndex===29 && $('status').textContent.startsWith('已加载')")
        b.js('loadFrame(12);loadFrame(0)')
        b.wait("currentFrameIndex===0 && $('status').textContent.startsWith('已加载')")
        assert b.js('stateCloud.snapshot.objects.length===13 && glState.drawCount===stateCloud.ids.length')
        checks['backward_and_rapid_switch']=True
        assert b.js('document.documentElement.scrollWidth<=innerWidth')
        b.call('Emulation.setDeviceMetricsOverride',dict(width=440,height=900,deviceScaleFactor=1,mobile=True))
        time.sleep(.3)
        assert b.js('document.documentElement.scrollWidth<=innerWidth')
        b.screenshot(qa/'mobile.png');checks['responsive']=True
        assert not b.exceptions,b.exceptions
        checks['runtime_exceptions']=b.exceptions;checks['passed']=True
    finally:
        if b:b.socket.close()
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close();(qa/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
