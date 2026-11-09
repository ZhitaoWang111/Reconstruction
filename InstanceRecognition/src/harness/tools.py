"""Evidence access and validated proposals, bounded by current cutoff."""
import json
from .state import propose
from .observations import model_observation
from .history import choose_representative, historical_bbox_view
from .reid_gate import verify_reidentification
from .view_params import ViewParams


def tool(name,description,properties):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{
        'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}
STRING={'type':'string'}
# `list_objects` was removed: the window's first message already carries this exact
# payload (`inputs.build_window_content` embeds `Tools.listing()`), and state cannot
# move until a commit, which ends the window. Calling it always returned what the
# model already had.
TOOLS=[tool('inspect_observations','Inspect up to 4 observations in full-frame RGB and highlighted overlays',{'ids':{'type':'array','items':STRING,'maxItems':4}}),
       tool('inspect_object','Inspect an archived object: metadata and one representative full-frame RGB (default) or all held observations. Pass all_observations=true when the object is full and you need to decide which observation to evict.',{'object_id':STRING,'all_observations':{'type':'boolean','default':False}}),
       tool('propose_update','Validate a proposal and return preview; does not commit',{'proposal_json':STRING}),
       tool('commit_update','Commit the checked proposal at current version',{'proposal_id':STRING})]



def _object_geometry(data, state, object_id):
    """Centroid and axis-aligned bounds from the object's held point cloud.

    Uses only held observations (status 'held' or pre-rename 'assigned'), so
    the geometry reflects what the harness actually committed. With at most five
    held observations the computation is fast enough for a tool call.
    Returns None when the object has no valid points.
    """
    held = [k for k, a in state['assignments'].items()
            if a.get('object_id') == object_id
            and a.get('status') in ('held', 'assigned')]
    if not held:
        return None
    import numpy as _np
    points_map = getattr(data, 'points', None)
    if points_map is None:
        return None
    parts = [points_map[k] for k in held if k in points_map and len(points_map[k])]
    if not parts:
        return None
    pts = _np.concatenate(parts)
    pts = pts[_np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return None
    centroid = pts.mean(axis=0)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    return {
        'centroid': [round(float(v), 3) for v in centroid],
        'bounds_min': [round(float(v), 3) for v in mn],
        'bounds_max': [round(float(v), 3) for v in mx],
        'size_xyz': [round(float(v), 3) for v in (mx - mn)],
        'geometry_scope': 'held observations only (max 5); axis-aligned',
    }

class Tools:
    def __init__(self,data,state,cutoff):
        self.data,self.state,self.cutoff=data,state,cutoff
        self.evidence=set();self.pending=None
        self.representatives={};self.sent_images=set()
        self.gate_relaxed=False
        _frames = getattr(data, 'source', None)
        self._vp = ViewParams(_frames / 'frames') if _frames else None

    def reidentification_checks(self,checked):
        """Holds that still need the archived object looked at, empty if none."""
        if self.gate_relaxed:return []
        centroids={oid:o['centroid'] for oid,o in self.data.items.items() if o.get('centroid')}
        report=verify_reidentification(checked['proposal']['actions'],self.state['assignments'],centroids)
        history={}
        for obs,record in self.state['assignments'].items():
            owner=record.get('object_id')
            if owner:history.setdefault(owner,set()).add(obs)
        held=[]
        for check in report['checks']:
            # inspect_object puts the representative observation into `evidence`,
            # which is also the record of what this window has shown the model.
            # Only historical observations count: current-window ids enter
            # `evidence` before any inspection happens.
            if history.get(check['object_id'],set())&self.evidence:continue
            held.append({**check,'history_count':len(history.get(check['object_id'],()))})
        return held

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
            # No archived observation id is listed here in any form. `observation_ids`
            # went first, then `representative_observation_id`: both are real ids a
            # model reads and cites, and the evidence check refuses them, because
            # reading a list is not inspecting. On the 500-frame run 25 of 26
            # "cites observations you have not inspected" errors quoted the
            # representative id verbatim. The count stays, because that is what the
            # decision needs; an id is reached through inspect_object, which records
            # it as evidence. The representative chosen for that call is still
            # written in full to window_inputs/*_context.json for audit.
            geo = _object_geometry(self.data, self.state, oid)
            centroid = geo['centroid'] if geo else None
            objects.append({'id':oid,**obj,'observation_count':len(ids),'centroid':centroid})
        # `pending` used to be re-sent here on every window: every assignment with
        # no owner, full record, `reason` and `evidence` included. Two things made
        # that dead weight. First, `exclude` is a closed decision - "this mask
        # spans two objects" never has a follow-up - so listing it as a to-do was
        # wrong on its face. Second, `defer` had no way back either: a deferral is
        # decided in the window that saw its mask, and that mask is never offered
        # again. Re-claiming one would mean assigning a historical observation by
        # id, and across the 500-frame run that happened 0 times out of 8,909
        # assigns, while 0 of 532 deferrals were ever reclaimed. The list therefore
        # only grew - 550 entries, 149 KB by the last window - and nothing read it.
        # Deferral stays a legal per-observation state (the coverage check in
        # state.propose requires every visible observation to have one); it is just
        # no longer broadcast. The records remain in state.json for audit.
        return {'version':self.state['version'],'objects':objects,'aliases':self.state['aliases']}

    def call(self,name,args):
        images=[]
        if name=='inspect_observations':return self.observe(args['ids'])
        if name=='inspect_object':
            obj=next((o for o in self.listing()['objects'] if o['id']==args['object_id']),None)
            if obj is None:raise ValueError('Unknown active object')
            object_id=args['object_id']
            all_obs=args.get('all_observations',False)
            # When the object is full or the caller explicitly wants all views,
            # return every held observation so the model can decide which to evict.
            held_ids=[i for i,a in self.state['assignments'].items()
                      if a.get('object_id')==object_id and
                         a.get('status') in ('held','assigned') and
                         self.data.items[i]['seq']<=self.state['cutoff']]
            from src.harness.state import HOLD_LIMIT
            return_all=all_obs or len(held_ids)>=HOLD_LIMIT
            if return_all:
                descriptors=[];reused=[];obs_meta=[]
                obj_c = None
                for o_id, o_obj in self.state['objects'].items():
                    if o_id == object_id:
                        members = [k for k,a in self.state['assignments'].items()
                                   if a.get('object_id')==o_id]
                        cents = [self.data.items[m].get('centroid') for m in members
                                 if m in self.data.items and self.data.items[m].get('centroid')]
                        if cents:
                            import numpy as _np
                            obj_c = _np.mean(cents, axis=0).tolist()
                        break
                for oid in held_ids:
                    path=historical_bbox_view(self.data,object_id,oid)
                    self.attach_image(path,{'observation_id':oid,'object_id':object_id,'kind':'historical_rgb_bbox'},images,descriptors,reused)
                    self.evidence.add(oid)
                    meta=dict(model_observation(self.data.items[oid]))
                    if self._vp is not None:
                        seq = self.data.items[oid]['seq']
                        vd = self._vp.view_direction(seq, obj_c)
                        cd = self._vp.camera_distance(seq, obj_c)
                        if vd is not None:
                            meta['view_direction'] = vd
                        if cd is not None:
                            meta['camera_distance_m'] = cd
                    obs_meta.append(meta)
                geo=_object_geometry(self.data,self.state,object_id)
                return {'object':obj,'object_id':object_id,
                        'observations':obs_meta,'held_count':len(held_ids),
                        'geometry':geo,
                        'note':'All held observations returned; cite the observation you intend to evict in the hold action evict field.',
                        'visual_images':descriptors,'reused_images':reused},images
            else:
                record=self.representative(object_id);oid=record['observation']['id']
                path=historical_bbox_view(self.data,object_id,oid)
                descriptors=[];reused=[]
                self.attach_image(path,{'observation_id':oid,'object_id':object_id,'kind':'historical_rgb_bbox'},images,descriptors,reused)
                self.evidence.add(oid)
                public_record=self.public_representative(object_id)
                geo=_object_geometry(self.data,self.state,object_id)
                return {'object':obj,**public_record,'observations':[public_record['observation']],
                        'geometry':geo,'visual_images':descriptors,'reused_images':reused},images
        if name=='propose_update':
            self.pending=None
            checked=propose(self.state,json.loads(args['proposal_json']),self.data.items,self.cutoff,self.evidence,self.data)
            self.pending=checked
            reid_checks=self.reidentification_checks(checked)
            if reid_checks:
                return {'status':'verification_required','proposal_id':checked['id'],
                        'proposal_held':True,
                        'note':'The proposal is valid and remains pending; it was not rejected. Before committing, call inspect_object for each object_id listed in checks and confirm the assignment against its archived appearance. Then resubmit the same proposal; resubmitting it unchanged is allowed, and so is changing the decision after inspecting.',
                        'checks':reid_checks,'base_version':self.state['version']},images
            # Post-proposal verify: back-projection and overlap checks.
            from .verify import build_checks, format_verify_message
            verify_checks, verify_images = build_checks(checked, self.data, self.state)
            fail_checks = [c for c in verify_checks if c.get('verify_fail')]
            if fail_checks:
                # attach_image drops images already sent this window into
                # `reused`; only the ones it keeps are returned, so the caller's
                # descriptor and image lists line up one for one.
                owner = {img: c for c in fail_checks for img in c['images']}
                attached, descriptors, reused = [], [], []
                for vimg in verify_images:
                    c = owner.get(str(vimg), {})
                    self.attach_image(vimg, {'kind': 'verify', 'object_id': c.get('object_id'),
                                             'observation_id': c.get('observation_id')},
                                      attached, descriptors, reused)
                return {'status': 'verify_required', 'proposal_id': checked['id'],
                        'verify_message': format_verify_message(verify_checks),
                        'checks': fail_checks,
                        'visual_images': descriptors, 'reused_images': reused,
                        'base_version': self.state['version']}, attached
            return {'proposal_id':checked['id'],'affected':checked['affected'],'object_count':len(checked['state']['objects']),
                    'actions':checked['proposal']['actions'],'base_version':self.state['version']},images
        if name=='commit_update':
            if self.pending is None or args['proposal_id']!=self.pending['id']:raise ValueError('No matching validated proposal')
            if self.pending['proposal']['base_version']!=self.state['version']:raise ValueError('Stale proposal')
            # Re-checked here as well: a proposal may be held, inspected, and then
            # skipped past with an older proposal_id if the model never resubmits.
            checks=self.reidentification_checks(self.pending)
            # The proposal is valid; what is missing is the look. Say that, or a
            # model reads this as a defect in the proposal and starts rewriting it.
            if checks:raise ValueError('Proposal accepted, not committed: these objects have not been inspected this window: '+
                ', '.join(sorted({c['object_id'] for c in checks}))+'. Call inspect_object on each, then commit again.')
            return {'ready_to_commit':True,'proposal_id':self.pending['id']},images
        raise ValueError('Unknown tool')
