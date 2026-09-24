"""Render exact redacted request structure with locally restored image evidence."""
import html
import json
import os
import sys
from pathlib import Path


def build(out):
    out=Path(out).resolve()
    observations=json.loads((out/'observations.json').read_text())
    chunks=['<!doctype html><meta charset="utf-8"><title>Harness v3 输入与调用记录</title><style>body{max-width:1100px;margin:36px auto;padding:20px;font:16px/1.7 system-ui;background:#f2f6f4;color:#193c34}details{background:white;padding:14px;margin:12px 0;border:1px solid #cddbd5;border-radius:8px}summary{cursor:pointer;font-weight:bold}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}a{color:#087d5d}</style><h1>第一个 5 帧窗口 · v3 实际输入与调用记录</h1><p>按帧组织 RGB、overlay 与观测信息；图片仅标类别。下方展示实际请求文本，图片根据保存的证据还原。密钥不在日志中。</p><p><a href="index.html">查看物体点云结果</a></p>']
    def details(title,value):
        return '<details><summary>'+html.escape(title)+'</summary><pre>'+html.escape(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'
    for req in sorted((out/'calls').glob('*_request.json')):
        prefix= req.name.split('_call')[0]
        round_number=int(req.name.split('_call')[1].split('_')[0])
        payload=json.loads(req.read_text())
        initial=[]
        for seq in range(5):
            o=next(o for o in observations.values() if o['seq']==seq)
            initial.extend([Path(o['image']),out/'evidence'/f'frame_{seq:04d}.jpg'])
        paths=initial[:]
        for log in sorted((out/'calls').glob(prefix+'_call*_tool*.json')):
            if int(log.name.split('_call')[1].split('_')[0])<round_number:
                paths.extend(Path(p) for p in json.loads(log.read_text())['images'])
        chunks.append('<h2>'+html.escape(req.stem)+'</h2>')
        chunks.append(details('请求参数与工具定义',{k:v for k,v in payload.items() if k!='messages'}))
        index=0
        for m,message in enumerate(payload['messages']):
            chunks.append('<details><summary>message '+str(m)+' · '+html.escape(message['role'])+'</summary>')
            content=message.get('content')
            if isinstance(content,list):
                for i,part in enumerate(content):
                    if part['type']=='image_url':
                        path=paths[index];index+=1
                        assert path.is_file(),path
                        chunks.append('<details><summary>content['+str(i)+'] · image</summary><img loading="lazy" src="'+html.escape(os.path.relpath(path,out),quote=True)+'"></details>')
                    else: chunks.append(details('content['+str(i)+'] · text',part))
            else:chunks.append(details('完整消息',message))
            chunks.append('</details>')
        assert index==len(paths),(index,len(paths))
    for log in sorted((out/'calls').glob('*_tool*.json')):
        chunks.append(details('工具执行 · '+log.stem,json.loads(log.read_text())))
    (out/'input_review.html').write_text('\n'.join(chunks))
    print(out/'input_review.html')

if __name__=='__main__':build(sys.argv[1])
