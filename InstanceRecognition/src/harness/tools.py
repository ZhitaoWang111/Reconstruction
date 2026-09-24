"""Evidence access and validated proposals, bounded by current cutoff."""
import json
import numpy as np
from scipy.spatial import cKDTree
from .state import propose
from .observations import model_observation
from .history import choose_representative, historical_bbox_view


def tool(name,description,properties):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{
        'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}
STRING={'type':'string'}
TOOLS=[tool('list_objects','List current objects and pending observations',{}),
       tool('inspect_observations','Inspect up to 4 observations in full-frame RGB and highlighted overlays',{'ids':{'type':'array','items':STRING,'maxItems':4}}),
       tool('inspect_object','Inspect an archived object: metadata and one full-frame RGB with its representative original-mask bbox; repeated images are reused',{'object_id':STRING}),
       tool('compare_geometry','Compare two observations; distances in meters are evidence, not identity rules',{'observation_a':STRING,'observation_b':STRING}),
       tool('propose_update','Validate a transaction and return preview; does not commit',{'proposal_json':STRING}),
       tool('commit_update','Commit the checked proposal at current version',{'proposal_id':STRING})]


class Tools:
    def __init__(self,data,state,cutoff):
        self.data,self.state,self.cutoff=data,state,cutoff
        self.evidence=set();self.pending=None;self.geometry={}
        self.representatives={};self.sent_images=set()

    def observe(self,ids):
        if not isinstance(ids,list) or not 1<=len(ids)<=4:raise ValueError('Request 1..4 observations')
        if any(i not in self.data.items or self.data.items[i]['seq']>self.cutoff for i in ids):
            raise ValueError('Unknown or future observation')
        images=[];descriptors=[];reused=[]
        for oid in ids:
            for kind,path in [('original_rgb',self.data.rgb_view(self.data.items[oid]['seq'])),
                              ('target_overlay',self.data.view(self.data.items[oid]['seq'],oid))]:
                self.attach_image(path, {'observation_id':oid,'kind':kind}, images, descriptors, reused)
        self.evidence.update(ids)
        return {'image_order':'For each observation: original RGB, then target overlay; already supplied images are omitted',
                'observations':[model_observation(self.data.items[i]) for i in ids],
                'visual_images':descriptors,'reused_images':reused},images

    def attach_image(self,path,descriptor,images,descriptors,reused):
        key=str(path)
        record={**descriptor,'image':key}
        if key in self.sent_images:
            reused.append(record)
        else:
            self.sent_images.add(key);images.append(path);descriptors.append(record)

    def representative(self,object_id):
        if object_id not in self.state['objects']:raise ValueError('Unknown active object')
        if object_id not in self.representatives:
            ids=[i for i,a in self.state['assignments'].items()
                 if a['object_id']==object_id and self.data.items[i]['seq']<=self.state['cutoff']]
            oid,metrics=choose_representative(self.data,ids)
            self.representatives[object_id]={'object_id':object_id,
                'observation':model_observation(self.data.items[oid]),'selection':metrics,
                'image_description':'one full-frame original RGB with selected original SAM3 mask bbox and Object ID; not necessarily full object extent'}
        return self.representatives[object_id]

    def public_representative(self, object_id):
        """Return only model-facing representative metadata; keep selection local."""
        record = self.representative(object_id)
        return {'object_id': record['object_id'], 'observation': record['observation']}

    def listing(self):
        objects=[]
        for oid,obj in self.state['objects'].items():
            ids=[i for i,a in self.state['assignments'].items() if a['object_id']==oid]
            representative=self.representatives.get(oid)
            objects.append({'id':oid,**obj,'observation_count':len(ids),'observation_ids':ids,
                            'representative_observation_id':representative['observation']['id'] if representative else None})
        return {'version':self.state['version'],'objects':objects,'aliases':self.state['aliases'],'pending':{k:v for k,v in self.state['assignments'].items() if v['object_id'] is None}}

    def call(self,name,args):
        images=[]
        if name=='list_objects':return self.listing(),images
        if name=='inspect_observations':return self.observe(args['ids'])
        if name=='inspect_object':
            obj=next((o for o in self.listing()['objects'] if o['id']==args['object_id']),None)
            if obj is None:raise ValueError('Unknown active object')
            record=self.representative(args['object_id']);oid=record['observation']['id']
            path=historical_bbox_view(self.data,args['object_id'],oid)
            descriptors=[];reused=[]
            self.attach_image(path,{'observation_id':oid,'object_id':args['object_id'],'kind':'historical_rgb_bbox'},images,descriptors,reused)
            self.evidence.add(oid)
            public_record=self.public_representative(args['object_id'])
            return {'object':obj,**public_record,'observations':[public_record['observation']],
                    'visual_images':descriptors,'reused_images':reused},images
        if name=='compare_geometry':
            a,b=args['observation_a'],args['observation_b']
            for i in (a,b):
                if i not in self.data.items or self.data.items[i]['seq']>self.cutoff:raise ValueError('Unknown or future observation')
            p,q=self.data.points[a][::8],self.data.points[b][::8]
            key=f'geometry:{a}:{b}'
            result={'id':key,'observations':[a,b],'note':'Subsampled observed surfaces, not visibility or proof of identity'}
            if len(p) and len(q):
                d=cKDTree(q).query(p)[0];e=cKDTree(p).query(q)[0]
                result.update(min_distance_m=float(d.min()),median_a_to_b_m=float(np.median(d)),median_b_to_a_m=float(np.median(e)),
                              a_fraction_within_5cm=float((d<.05).mean()),b_fraction_within_5cm=float((e<.05).mean()))
            else:result['unavailable']='No valid points'
            self.evidence.add(key);self.geometry[key]=result;return result,images
        if name=='propose_update':
            self.pending=None
            checked=propose(self.state,json.loads(args['proposal_json']),self.data.items,self.cutoff,self.evidence)
            self.pending=checked
            return {'proposal_id':checked['id'],'affected':checked['affected'],'object_count':len(checked['state']['objects']),
                    'pending_count':sum(a['object_id'] is None for a in checked['state']['assignments'].values()),
                    'actions':checked['proposal']['actions'],'base_version':self.state['version']},images
        if name=='commit_update':
            if self.pending is None or args['proposal_id']!=self.pending['id']:raise ValueError('No matching validated proposal')
            if self.pending['proposal']['base_version']!=self.state['version']:raise ValueError('Stale proposal')
            return {'ready_to_commit':True,'proposal_id':self.pending['id']},images
        raise ValueError('Unknown tool')
