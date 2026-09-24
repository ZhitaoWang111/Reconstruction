"""Export saved logs verbatim as JSON values; do not run models or tools."""
import json,re,hashlib,os
from pathlib import Path
root=Path(__file__).resolve().parents[3];out=Path(__file__).parent
runs=[]
for name in ['harness_window01_v3','harness_first30_v1']:
 folder=root/'InstanceRecognition/runtime/experiments'/name
 windows={}
 for f in sorted((folder/'calls').glob('*.json')):
  m=re.fullmatch(r'w(\d+)_(\d+)_call(\d+)_(request|tool\d+|\d+_(?:response|meta|error))\.json',f.name)
  if not m:raise ValueError(f'Unrecognized log: {f}')
  start,session,call,kind=m.groups();win=windows.setdefault(int(start),{})
  step=win.setdefault(session,{}).setdefault(int(call),[])
  raw=f.read_text();step.append({'kind':kind,'file':str(f.relative_to(root)),'url':os.path.relpath(f,root/'docs'),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'data':json.loads(raw)})
 def rank(e):
  k=e['kind']
  if k=='request':return (0,0,0)
  if k.startswith('tool'):return (2,int(k[4:]),0)
  a,t=k.split('_');return(1,int(a),{'response':0,'meta':1,'error':2}[t])
 config=json.loads((folder/'config.json').read_text())
 runs.append({'name':name,'config':config,'windows':[{'start':w,'sessions':[{'id':sid,'calls':[{'index':idx,'events':sorted(events,key=rank)} for idx,events in sorted(calls.items())]} for sid,calls in sorted(sessions.items())]} for w,sessions in sorted(windows.items())]})
(out/'records.js').write_text('window.AGENT_RECORDS='+json.dumps(runs,ensure_ascii=False).replace('</','<\\/')+';\n')
print('Exported',sum(len(c['events']) for r in runs for w in r['windows'] for s in w['sessions'] for c in s['calls']),'original log files')
