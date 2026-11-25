#!/usr/bin/env python3
"""Validate exported ownership against ScanNet annotations and exercise file:// UI."""
import json,sys,tempfile,time,subprocess,urllib.request
from pathlib import Path
import numpy as np
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'InstanceRecognition/tests'))
from check_mask_pointcloud_viewer import Browser
OUT=ROOT/'ScanNet/assets/object_gt'

def main():
    qa=OUT/'qa';qa.mkdir(exist_ok=True);checks={'scenes':[]}
    for folder in sorted(OUT.glob('scene*')):
        meta=json.loads((folder/'manifest.json').read_text());groups=json.loads(Path(meta['sources']['aggregation']['path']).read_text())['segGroups']
        segments=np.array(json.loads(Path(meta['sources']['segments']['path']).read_text())['segIndices'])
        vertices=PlyData.read(meta['sources']['mesh']['path'])['vertex'];original=np.column_stack([vertices[k] for k in ('x','y','z')])
        scene=np.load(folder/'scene.npz');expected_owner=np.full(len(vertices),-1)
        for g in groups:
            expected=np.flatnonzero(np.isin(segments,g['segments']));oid=g['objectId'];expected_owner[expected]=oid
            with np.load(folder/f'objects/object_{oid:04d}.npz') as obj:
                assert np.array_equal(obj['vertex_indices'],expected)
                assert np.array_equal(obj['points_scan'],original[expected])
                assert np.array_equal(obj['segment_ids'],segments[expected])
                expected_world=(np.column_stack([original[expected],np.ones(len(expected))])@np.array(meta['axis_alignment']).T)[:,:3]
                assert np.allclose(obj['points_world'],expected_world,atol=1e-6)
                ply=PlyData.read(folder/f'objects/object_{oid:04d}.ply')['vertex']
                assert len(ply)==len(expected) and np.array_equal(ply['vertex_index'],expected)
                assert np.all(ply['object_id']==oid)
        assert np.array_equal(scene['object_ids'],expected_owner)
        checks['scenes'].append({'scene':folder.name,'objects':len(groups),'vertices':len(vertices),'exact_membership':True})
        scene.close()
    profile=Path(tempfile.mkdtemp(prefix='object-gt-chrome-'));log=(qa/'chrome.log').open('wb')
    proc=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--disable-extensions','--no-first-run','--enable-unsafe-swiftshader','--use-angle=swiftshader','--remote-debugging-port=0',f'--user-data-dir={profile}','about:blank'],stdout=log,stderr=log);b=None
    try:
        deadline=time.monotonic()+30
        while not (profile/'DevToolsActivePort').exists() and time.monotonic()<deadline:time.sleep(.1)
        port=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as f:targets=json.load(f)
        b=Browser(next(t for t in targets if t['type']=='page')['webSocketDebuggerUrl']);b.call('Runtime.enable');b.call('Page.enable')
        b.call('Emulation.setDeviceMetricsOverride',dict(width=1600,height=1150,deviceScaleFactor=1,mobile=False))
        b.call('Page.navigate',dict(url=(ROOT/'ScanNet/objects/index.html').as_uri()))
        b.wait("typeof gtScene!=='undefined' && gtScene && $('status').textContent.startsWith('已加载')")
        assert b.js('gtScene.meta.objects.length')==67
        assert b.js('glState.gl.getError()')==0
        b.screenshot(qa/'scene0000_00.png')
        # Validate every object's exact selected vertex set, including ID zero.
        for oid in b.js('gtScene.meta.objects.map(o=>o.object_id)'):
            b.js(f'gtSelection.clear();chooseGT({oid})')
            assert b.js(f'visibleVertices.length===gtScene.byId.get({oid}).vertex_count && Array.from(visibleVertices).every(i=>gtScene.owners[i]==={oid})')
            assert b.js('Array.from(visibleFaces).every(i=>gtSelection.has(gtScene.owners[i]))')
        checks['all_67_object_selections']=True
        oid=b.js("gtScene.meta.objects.find(o=>o.label==='toilet').object_id")
        b.js(f"gtSelection.clear();chooseGT({oid});$('focus').click();$('showBackground').checked=false;renderCloud()")
        b.screenshot(qa/'toilet.png')
        count=b.js('visibleVertices.length');assert count>0
        b.js("$('modePoints').click();$('colorMode').value='rgb';$('colorMode').dispatchEvent(new Event('change'))")
        assert b.js('visibleVertices.length')==count;b.screenshot(qa/'toilet_rgb_points.png')
        b.js("$('modeMesh').click();$('colorMode').value='segment';$('colorMode').dispatchEvent(new Event('change'))")
        assert b.js('visibleVertices.length')==count
        # Actual pixel evidence: non-background fragments must be present.
        checks['colored_pixels']=b.js("""(()=>{renderCloud();const gl=glState.gl,p=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,p);let n=0;for(let i=0;i<p.length;i+=4)if(p[i]>30||p[i+1]>30||p[i+2]>30)n++;return n;})()""")
        assert checks['colored_pixels']>100
        # Category/search/multiselect affect data, not only row labels.
        b.js("$('category').value='toilet';$('category').dispatchEvent(new Event('change'))")
        assert b.js('visibleVertices.length')==count
        b.js("$('search').value='nonexistent';$('search').dispatchEvent(new Event('input'))")
        assert b.js('visibleVertices.length===0 && visibleFaces.length===0')
        b.js("$('search').value='';$('search').dispatchEvent(new Event('input'))")
        # Click a triangle center in the isolated toilet category to select it.
        xy=b.js("""(()=>{const c=$('cloudCanvas'),r=c.getBoundingClientRect(),m=computeMatrix(c.width/c.height),v=visibleFaces.slice(0,3),p=[0,1,2].map(k=>v.reduce((s,i)=>s+gtScene.positions[i*3+k],0)/3),a=[...p,1],q=[0,1,2,3].map(k=>a.reduce((s,x,j)=>s+x*m[j*4+k],0));return [r.left+(q[0]/q[3]+1)*r.width/2,r.top+(1-q[1]/q[3])*r.height/2];})()""")
        b.mouse('mousePressed',*xy,button='left',buttons=1);b.mouse('mouseReleased',*xy,button='left',buttons=0)
        assert b.js(f'gtSelection.has({oid})');checks['surface_picking']=True
        b.js("$('category').value='';$('category').dispatchEvent(new Event('change'));chooseGT(0);chooseGT(1,true)")
        assert b.js('gtSelection.size===2 && visibleVertices.length===gtScene.byId.get(0).vertex_count+gtScene.byId.get(1).vertex_count')
        for sid in ['scene0000_02','scene0001_01','scene0000_00']:
            b.js(f"loadGTScene('{sid}')");assert b.js('glState.gl.getError()')==0
        b.js("loadGTScene('scene0000_02');loadGTScene('scene0000_00')")
        b.wait("gtScene.meta.scene_id==='scene0000_00' && $('status').textContent.startsWith('已加载')")
        checks['scene_switching']=True
        b.js("$('category').value='shoes';$('category').dispatchEvent(new Event('change'));chooseGT(filteredObjects()[0].object_id);$('focus').click();$('colorMode').value='instance';updateGT()")
        b.screenshot(qa/'shoes.png')
        assert b.js('document.documentElement.scrollWidth<=innerWidth')
        b.call('Emulation.setDeviceMetricsOverride',dict(width=440,height=900,deviceScaleFactor=1,mobile=True));time.sleep(.3)
        assert b.js('document.documentElement.scrollWidth<=innerWidth');b.screenshot(qa/'mobile.png')
        assert not b.exceptions,b.exceptions
        checks['runtime_exceptions']=b.exceptions;checks['responsive']=True;checks['passed']=True
    finally:
        if b:b.socket.close()
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close();(qa/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
