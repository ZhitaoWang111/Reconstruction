"""Association invariants, independent of ScanNet labels and trained models."""
import unittest
import numpy as np
from scipy.spatial import cKDTree
from src.geometry.incremental_objects import Associator, voxelize


def patch(x=0):
    return np.array([[x+.01+i*.03,.01+j*.03,1.01] for i in range(8) for j in range(8)],np.float32)


def observation(seq,index,points,category_label='chair',pixels=None):
    colors=np.full(points.shape,128,np.uint8)
    return dict(id=f'{seq}:{index}',seq=seq,mask_index=index,instance_id=index+1,
        category_label=category_label,points=voxelize(points,colors,.02)[1],raw_points=points,
        raw_colors=colors,pixels=np.arange(len(points)) if pixels is None else pixels)


def surface(observations):
    # Duplicate pixel coordinates correctly carry every corresponding mask bit.
    points=np.concatenate([o['raw_points'] for o in observations])
    points=np.unique(points,axis=0);bits=np.zeros(len(points),np.uint32)
    tree=cKDTree(points)
    for o in observations:bits[tree.query(o['raw_points'])[1]] |= np.uint32(1<<o['mask_index'])
    return tree,bits


class IncrementalTests(unittest.TestCase):
    def test_alias_identity_and_distinct_same_class(self):
        a=Associator()
        for seq in range(3):
            obs=[observation(seq,0,patch(), 'office chair' if seq==0 else 'chair'),
                 observation(seq,1,patch(1),pixels=np.arange(64)+64)]
            a.step(seq,obs,*surface(obs))
            self.assertEqual([o['object_id'] for o in obs],[1,2])
        self.assertEqual(len(a.objects),2)
        self.assertTrue((a.snapshot()[1]['support_frame_bits']==7).all())

    def test_duplicate_masks_do_not_double_support(self):
        a=Associator();obs=[observation(0,0,patch())];a.step(0,obs,*surface(obs))
        obs=[observation(1,i,patch()) for i in range(2)]
        a.step(1,obs,*surface(obs))
        self.assertEqual([o['object_id'] for o in obs],[1,1])
        self.assertTrue(all(r['count']==2 for r in a.objects[1]['voxels'].values()))
        self.assertEqual(a.objects[1]['labels']['chair'],2)

    def test_bridge_is_pending_and_absence_preserves_objects(self):
        a=Associator();obs=[observation(0,0,patch()),observation(0,1,patch(1))]
        a.step(0,obs,*surface(obs))
        obs=[observation(1,0,np.concatenate([patch(),patch(1)]))]
        a.step(1,obs,*surface(obs))
        self.assertIsNone(obs[0]['object_id']);self.assertEqual(len(a.objects),2)
        a.step(2,[],*surface(obs));self.assertEqual(len(a.objects),2)

    def test_pair_does_not_absorb_individual(self):
        a=Associator();obs=[observation(0,0,patch(),'shoe')];a.step(0,obs,*surface(obs))
        obs=[observation(1,0,patch(),'pair of shoes')];a.step(1,obs,*surface(obs))
        self.assertEqual(obs[0]['object_id'],2)

    def test_deferred_points_have_observation_and_insertion_times(self):
        a=Associator();obs=[observation(0,0,patch())];a.step(0,obs,*surface(obs))
        historical=observation(0,1,patch(1),pixels=np.arange(64)+64);a.register(historical)
        a.current=2;a.objects[1]['observations'].append(historical['id']);a.rebuild(1)
        data=a.snapshot()[1]
        self.assertTrue((data['first_seen']==0).all())
        self.assertEqual(set(data['added_at']),{0,2})

    def test_reject_out_of_order(self):
        with self.assertRaises(ValueError):Associator().step(1,[],None,None)


if __name__=='__main__':unittest.main()
