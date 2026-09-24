"""Bounded gateway requests with redacted logs and complete-response validation."""
import base64
import json
import time
import urllib.request
import urllib.error
from src.inference.gpt_labels import client_module
from .state import save


def image(path):
    return {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode(),'detail':'high'}}


def redacted(value, key):
    if isinstance(value,str):
        if value.startswith('data:image/'):return '[image bytes stored in evidence/]'
        return value.replace(key,'[REDACTED]') if key else value
    if isinstance(value,list):return [redacted(v,key) for v in value]
    if isinstance(value,dict):return {k:redacted(v,key) for k,v in value.items()}
    return value


def read_stream(response):
    """Assemble SSE tool-call fragments; truncated streams never execute."""
    calls={};body={'choices':[]};content='';finish=None;done=False
    for raw in response:
        line=raw.decode('utf-8').strip()
        if not line.startswith('data:'):continue
        data=line[5:].strip()
        if data=='[DONE]':done=True;break
        chunk=json.loads(data)
        for k in ('id','model','usage'):
            if chunk.get(k) is not None:body[k]=chunk[k]
        for choice in chunk.get('choices',[]):
            if choice.get('finish_reason'):finish=choice['finish_reason']
            delta=choice.get('delta',{});content+=delta.get('content') or ''
            for c in delta.get('tool_calls',[]):
                item=calls.setdefault(c['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                if c.get('id'):item['id']+=c['id']
                for k in ('name','arguments'):item['function'][k]+=c.get('function',{}).get(k) or ''
    if not done or finish is None:raise ValueError('Incomplete SSE stream')
    body['choices']=[{'finish_reason':finish,'message':{'role':'assistant','content':content or None,'tool_calls':[calls[i] for i in sorted(calls)]}}]
    return body


class Client:
    def __init__(self, model, out, timeout=120):
        cfg=client_module().settings();self.key=cfg.get('VIBEG_API_KEY','')
        if not self.key:raise RuntimeError('Missing VIBEG_API_KEY')
        self.base=cfg.get('VIBEG_OPENAI_BASE_URL','https://vibeg.seer-robotics.cn/openai/v1').rstrip('/')
        self.model,self.out,self.timeout=model,out,timeout

    def call(self,messages,tools,tag,final=False):
        payload={'model':self.model,'reasoning_effort':'none','max_completion_tokens':10000,
                 'messages':messages,'tools':tools,'parallel_tool_calls':False,'stream':True,'stream_options':{'include_usage':True},
                 'tool_choice':{'type':'function','function':{'name':'propose_update'}} if final else 'required'}
        save(self.out/'calls'/f'{tag}_request.json',redacted(payload,self.key))
        for attempt in range(3):
            start=time.monotonic()
            request=urllib.request.Request(self.base+'/chat/completions',json.dumps(payload).encode(),
                headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(request,timeout=self.timeout) as r:
                    body=read_stream(r) if 'text/event-stream' in r.headers.get('Content-Type','') else json.loads(r.read())
                save(self.out/'calls'/f'{tag}_{attempt}_response.json',redacted(body,self.key))
                choice=body.get('choices',[{}])[0];message=choice.get('message',{})
                if choice.get('finish_reason') not in ('stop','tool_calls') or not message.get('tool_calls'):
                    raise ValueError('Missing complete tool call response')
                save(self.out/'calls'/f'{tag}_{attempt}_meta.json',{'elapsed_seconds':time.monotonic()-start,'usage':body.get('usage'),
                    'requested_model':self.model,'returned_model':body.get('model'),'response_id':body.get('id')})
                return message
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                if isinstance(exc,urllib.error.HTTPError):error+= ': '+exc.read().decode(errors='replace')[:1800]
                save(self.out/'calls'/f'{tag}_{attempt}_error.json',{'error':redacted(error,self.key),'seconds':time.monotonic()-start})
                if attempt==2:raise RuntimeError(redacted(error,self.key)) from None
                time.sleep(2**attempt)
