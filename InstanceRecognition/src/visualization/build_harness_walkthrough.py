"""Build an offline first-window walkthrough from committed Harness evidence."""
import json
import base64
import hashlib
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'InstanceRecognition/runtime/experiments/harness_first30_v1'


def build(out=OUT):
    transaction = json.loads((out/'versions/0001/transaction.json').read_text())
    state = transaction['state']
    observations = json.loads((out/'observations.json').read_text())
    observations = {k:v for k,v in observations.items() if v['seq'] <= state['cutoff']}
    commits = []
    for f in sorted((out/'calls').glob('w0000*_tool*.json')):
        d = json.loads(f.read_text())
        if d['name'] == 'commit_update' and d['result'].get('committed') == 1:
            commits.append(f)
    if len(commits) != 1:
        raise ValueError('Expected exactly one successful first-window commit')
    prefix = commits[0].name.split('_call')[0]
    calls = []
    for f in sorted((out/'calls').glob(prefix+'_call*_tool*.json')):
        d = json.loads(f.read_text())
        d['arguments'] = json.loads(d['arguments'])
        d['images'] = [os.path.relpath(p, out) for p in d['images']]
        d['round'] = int(f.name.split('_call')[1].split('_')[0])+1
        d['log'] = f'calls/{f.name}'
        calls.append(d)
    request_file = out/'calls'/f'{prefix}_call00_request.json'
    request = json.loads(request_file.read_text())
    # Reconstruct all actual request messages; extract exact image bytes for offline display.
    requests = []
    assets = out/'request_images'
    assets.mkdir(exist_ok=True)
    for request_path in sorted((out/'calls').glob(prefix+'_call*_request.json')):
        payload = json.loads(request_path.read_text())
        image_paths = [out/'evidence'/f'frame_{i:04d}.jpg' for i in range(5)]
        request_round = int(request_path.name.split('_call')[1].split('_')[0])+1
        image_paths += [out/image for call in calls if call['round'] < request_round for image in call['images']]
        image_index = 0
        for message in payload['messages']:
            if not isinstance(message.get('content'), list):
                continue
            for part in message['content']:
                if part['type'] != 'image_url':
                    continue
                url = part['image_url']['url']
                if url == '[image bytes stored in evidence/]':
                    source_image = image_paths[image_index]
                    raw = source_image.read_bytes()
                    header = 'data:image/jpeg;base64'
                else:
                    header, encoded = url.split(',', 1)
                    raw = base64.b64decode(encoded, validate=True)
                image_index += 1
                sha = hashlib.sha256(raw).hexdigest()
                suffix = '.jpg' if 'jpeg' in header else '.png'
                path = assets/(sha+suffix)
                path.write_bytes(raw)
                part['image_url']['url'] = 'request_images/'+path.name
                part['transport'] = {'original_type': 'base64 data URL', 'mime': header[5:].split(';')[0],
                                     'bytes': len(raw), 'sha256': sha, 'provenance': '日志省略图片字节；按消息顺序和工具记录关联到保存的 evidence 文件'}
        if image_index != len(image_paths):
            raise ValueError('Evidence/message image count mismatch')
        requests.append({'log': 'calls/'+request_path.name, 'payload': payload})
    clouds = {}
    for oid in state['objects']:
        with np.load(out/f'versions/0001/clouds/{oid}.npz') as z:
            p=z['points']; step=max(1,(len(p)+1599)//1600)
            clouds[oid]={'count':len(p),'points':p[::step].round(4).tolist()}
    for oid,o in observations.items():
        o['image'] = os.path.relpath(o['image'],out)
        o['mask'] = os.path.relpath(o['mask'],out)
        o['crop'] = 'evidence/'+oid.replace(':','_')+'.jpg'
        o['point_file'] = os.path.relpath(o['point_file'],ROOT)
    data={'observations':observations,'state':state,'transaction':transaction,'calls':calls,
          'requests':requests,'clouds':clouds,'run_prefix':prefix,
          'timing':json.loads((out/'versions/0001/timing.json').read_text()),
          'config':json.loads((out/'config.json').read_text())}
    (out/'window01_walkthrough.js').write_text('window.WALKTHROUGH='+json.dumps(data,ensure_ascii=False).replace('</','<\\/')+';\n')
    template=Path(__file__).with_name('harness_walkthrough.html')
    (out/'window01_walkthrough.html').write_text(template.read_text())
    print(out/'window01_walkthrough.html')

if __name__=='__main__':
    build()
