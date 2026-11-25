"""M5 routing/recovery checks without GPU or network."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objgen.appearance.contracts import atomic_json,digest
from objgen.appearance.orchestrator import verified_delivery,selected_asset,execute


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name);self.rev=self.run/'usd/appearance/v_test';self.rev.mkdir(parents=True)
        self.folder=self.run/'physics/appearance/v_test/m4_test';(self.folder/'asset/textures').mkdir(parents=True)
        (self.run/'physics/object.usda').write_text('baseline');(self.rev/'object.usda').write_text('visual')
        (self.folder/'asset/object.usda').write_text('final');(self.folder/'asset/textures/color.png').write_bytes(b'pixels')
        self.snapshot={'complete':True,'revision':'v_test','fingerprint':'fixed'}
        p=patch('objgen.appearance.orchestrator.candidate_snapshot',return_value=self.snapshot);p.start();self.addCleanup(p.stop)
        self.report={'status':'success','source_revision':'v_test','candidate_sha256':digest(self.rev/'object.usda'),
                     'baseline_physics_sha256':digest(self.run/'physics/object.usda'),'candidate_snapshot':self.snapshot,
                     'output_sha256':{str(p.relative_to(self.folder)):digest(p) for p in self.folder.rglob('*') if p.is_file()}}
        self.report.update({k:True for k in ['passed','physics_preserved','isaac_display_verified','drop_verified','texture_sampled']})
        atomic_json(self.folder/'report.json',self.report)
    def select(self):
        record=verified_delivery(self.run,self.folder,'v_test')
        atomic_json(self.run/'usd/appearance/delivery_selection.json',{'mode':'technical_verified','human_approved':False,'delivery':str(self.folder),'revision':'v_test','verification':record})
    def test_selection_detects_texture_tampering(self):
        self.select();self.assertEqual(selected_asset(self.run),self.folder/'asset/object.usda')
        (self.folder/'asset/textures/color.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):selected_asset(self.run)
    def test_requires_all_checks_and_matching_revision(self):
        with self.assertRaises(ValueError):verified_delivery(self.run,self.folder,'other')
        self.report['drop_verified']=False;atomic_json(self.folder/'report.json',self.report)
        with self.assertRaises(ValueError):verified_delivery(self.run,self.folder)
    def test_resume_does_not_rerun_delivery(self):
        inputs={'physics_object.usda':{'path':str(self.run/'physics/object.usda')}}
        blender=self.run/'blender';blender.write_text('binary');env=self.run/'env.sh';env.write_text('environment')
        with patch('objgen.appearance.orchestrator.resolve_inputs',return_value=inputs),patch('objgen.appearance.orchestrator.deliver',return_value=self.folder) as deliver:
            folder,report=execute(self.run,{'blender':str(blender)},env,'v_test',feedback=False)
            self.assertEqual(report['status'],'success');self.assertEqual(deliver.call_count,1)
            _,resumed=execute(self.run,{'blender':str(blender)},env,'v_test',resume=folder,feedback=False)
            self.assertEqual(resumed['asset'],report['asset']);self.assertEqual(deliver.call_count,1)
            blender.write_text('new binary')
            with self.assertRaisesRegex(ValueError,'Recovery'):execute(self.run,{'blender':str(blender)},env,'v_test',resume=folder,feedback=False)


if __name__=='__main__':unittest.main()
