"""Conservative incremental association of immutable mask observations.

Surface support is measured against the current point map; absence is unknown,
not negative visibility evidence. This prototype does not infer hidden volume.
"""
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree


def family(label):
    aliases = {'papers':'paper','books':'book','curtains':'curtain','office chair':'chair',
        'desk':'table','low table':'table','side table':'table','mobile phone':'phone',
        'smartphone':'phone','wall clock':'clock','fabric wardrobe':'wardrobe',
        'trash can':'bin','trash bin':'bin','wastebasket':'bin','computer mouse':'mouse',
        'wall-mounted coat rack':'coat rack','desk lamp':'lamp','shoes':'shoe'}
    return aliases.get(label,label)


def granularity(label):
    return 'semantic_group' if label.startswith('pair of ') else 'object_candidate'


def voxelize(points, colors, size):
    keys, inverse = np.unique(np.floor(points/size).astype(np.int32), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    xyz = np.column_stack([np.bincount(inverse,weights=points[:,i])/counts for i in range(3)])
    rgb = np.column_stack([np.bincount(inverse,weights=colors[:,i])/counts for i in range(3)])
    return keys,xyz.astype(np.float32),rgb.astype(np.float32)


class Associator:
    def __init__(self, voxel=.02, distance=.05, margin=.12):
        self.voxel,self.distance,self.margin=voxel,distance,margin
        self.objects={};self.observations={};self.pending=[];self.events=[]
        self.current=-1;self.next_id=1;self.surface_cache={}

    def register(self, observation):
        self.observations[observation['id']]=observation

    def object_points(self,obj):
        rows=list(obj['voxels'].values())
        return np.array([r['xyz']/r['count'] for r in rows],dtype=np.float32).reshape(-1,3)

    def candidates(self,obs,surface_tree,surface_bits):
        candidates=[];p=obs['points'];lo=p.min(0);hi=p.max(0)
        for oid,obj in self.objects.items():
            if obj['kind']!=granularity(obs['category_label']):continue
            q=obj['points']
            if np.any(hi+self.distance<q.min(0)) or np.any(q.max(0)+self.distance<lo):continue
            dist=obj['tree'].query(p,workers=1)[0];forward=float(np.mean(dist<=self.distance))
            if forward<.12:continue
            key=(oid,obs['seq'],len(obj['observations']))
            if key not in self.surface_cache:
                d,idx=surface_tree.query(q,workers=1)
                self.surface_cache[key]=(d<=self.distance,idx)
            shared,idx=self.surface_cache[key];n=int(shared.sum())
            reverse=float(np.mean((surface_bits[idx[shared]] & np.uint32(1<<obs['mask_index']))!=0)) if n else 0.
            compatible=family(obs['category_label']) in {family(x) for x in obj['labels']}
            score=.55*forward+.45*reverse
            strong=n>=12 and ((forward>=.65 and reverse>=.55) or (forward>=.88 and reverse>=.20))
            # Different semantic labels need near-duplicate surface coverage.
            if not compatible:strong=strong and forward>=.90 and reverse>=.90
            candidates.append({'object_id':oid,'forward':round(forward,4),'reverse':round(reverse,4),
                'shared_voxels':n,'compatible':compatible,'score':round(score,4),'strong':bool(strong)})
        return sorted(candidates,key=lambda c:(-c['score'],c['object_id']))

    def choose(self,candidates):
        if not candidates:return 'new',None
        top=candidates[0]
        # A mask spanning several established surfaces is a bridge, not an
        # instruction to union object identities.
        bridges=[c for c in candidates if c['reverse']>=.55 and c['forward']>=.2]
        if len(bridges)>1:return 'pending',None
        if top['strong'] and (len(candidates)==1 or top['score']-candidates[1]['score']>=self.margin):
            return 'match',top['object_id']
        if top['forward']<.25 and top['reverse']<.25:return 'new',None
        return 'pending',None

    def rebuild(self,oid):
        obj=self.objects[oid];by_frame=defaultdict(list)
        for key in obj['observations']:by_frame[self.observations[key]['seq']].append(self.observations[key])
        voxels={};labels=Counter()
        for seq,observations in sorted(by_frame.items()):
            # Original pixel index identifies duplicate memberships in a frame.
            pixels=np.concatenate([o['pixels'] for o in observations]);_,take=np.unique(pixels,return_index=True)
            points=np.concatenate([o['raw_points'] for o in observations])[take]
            colors=np.concatenate([o['raw_colors'] for o in observations])[take]
            keys,xyz,rgb=voxelize(points,colors,self.voxel)
            for k,p,c in zip(keys,xyz,rgb):
                key=tuple(k);r=voxels.setdefault(key,{'xyz':np.zeros(3),'rgb':np.zeros(3),'count':0,'frames':0,'first':seq,
                    'added':obj['voxels'].get(key,{}).get('added',self.current)})
                r['xyz']+=p;r['rgb']+=c;r['count']+=1;r['frames']|=1<<seq;r['first']=min(r['first'],seq)
            # Same-frame aliases or duplicate masks cannot stack semantic votes.
            labels.update(set(o['category_label'] for o in observations))
        obj.update(voxels=voxels,labels=labels,frames=sorted(by_frame))
        obj['points']=self.object_points(obj);obj['tree']=cKDTree(obj['points'])
        self.surface_cache={k:v for k,v in self.surface_cache.items() if k[0]!=oid}

    def step(self,seq,observations,surface_tree,surface_bits):
        if seq!=self.current+1:raise ValueError('Process consecutive sampled sequences')
        self.current=seq;self.surface_cache={};touched=set()
        # Match against the previous state; no order-dependent same-frame update.
        decisions=[]
        for obs in observations:
            self.register(obs);c=self.candidates(obs,surface_tree,surface_bits)
            action,oid=self.choose(c);decisions.append((obs,c,action,oid))
        for obs,c,action,oid in decisions:
            if action=='new':
                oid=self.next_id;self.next_id+=1
                self.objects[oid]={'id':oid,'created_at':seq,'kind':granularity(obs['category_label']),
                    'observations':[],'voxels':{},'labels':Counter()}
            obs.update(object_id=oid,status=action,candidates=c[:4])
            if oid is not None:self.objects[oid]['observations'].append(obs['id']);touched.add(oid)
            else:self.pending.append(obs['id'])
            self.events.append({'at':seq,'observation':obs['id'],'action':action,'object_id':oid,'candidates':c[:4]})
        for oid in sorted(touched):self.rebuild(oid)

    def retry_pending(self,surfaces):
        changed=set();remaining=[]
        for key in self.pending:
            obs=self.observations[key]
            if obs['seq']==self.current:remaining.append(key);continue
            tree,bits=surfaces[obs['seq']]
            c=self.candidates(obs,tree,bits);action,oid=self.choose(c)
            if action=='match' and len(self.objects[oid]['frames'])>=2:
                obs.update(object_id=oid,status='resolved',candidates=c[:4])
                self.objects[oid]['observations'].append(key);changed.add(oid)
                self.events.append({'at':self.current,'observation':key,'action':'resolve_pending','object_id':oid,'candidates':c[:4]})
            else:remaining.append(key)
        self.pending=remaining
        for oid in sorted(changed):self.rebuild(oid)

    def snapshot(self):
        objects=[];positions=[];colors=[];ids=[];supports=[];first=[];added=[];conflicts=[]
        owners=Counter(k for obj in self.objects.values() for k in obj['voxels'])
        for oid,obj in sorted(self.objects.items()):
            rows=list(obj['voxels'].items());multi=sum(r['count']>=3 for _,r in rows)
            label=obj['labels'].most_common(1)[0][0]
            objects.append({'id':oid,'object_label':label,'labels':dict(obj['labels']),'kind':obj['kind'],
                'created_at':obj['created_at'],'frames':obj['frames'],'observations':list(obj['observations']),
                'voxel_count':len(rows),'multi_frame_voxels':multi,'status':'multi_frame' if len(obj['frames'])>=3 else 'tentative'})
            for key,r in rows:
                positions.append(r['xyz']/r['count']);colors.append(r['rgb']/r['count']);ids.append(oid)
                supports.append(r['frames']);first.append(r['first']);added.append(r['added']);conflicts.append(owners[key]>1)
        return objects,{'points_world':np.asarray(positions,dtype=np.float32).reshape(-1,3),
            'rgb':np.asarray(colors,dtype=np.uint8).reshape(-1,3),'object_ids':np.asarray(ids,dtype=np.uint32),
            'support_frame_bits':np.asarray(supports,dtype=np.uint32),'first_seen':np.asarray(first,dtype=np.uint8),
            'added_at':np.asarray(added,dtype=np.uint8),'ownership_conflict':np.asarray(conflicts,dtype=np.uint8)}
