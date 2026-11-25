import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from objgen.appearance.contracts import atomic_json,digest
from objgen.appearance.review import candidate_snapshot,decide,verified_selection
from objgen.appearance.feedback import reserve,validate_action


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name);self.rev=self.run/'usd/appearance/v_test';self.rev.mkdir(parents=True)
        self.inputs={'source':{'sha256':'fixed'}}
        self.patcher=patch('objgen.appearance.review.resolve_inputs',return_value=self.inputs);self.patcher.start();self.addCleanup(self.patcher.stop)
        atomic_json(self.rev/'manifest.json',{'inputs':self.inputs});atomic_json(self.rev/'config.json',{})
        (self.rev/'object.usda').write_text('candidate')
        atomic_json(self.rev/'report.json',{'technical_checks_passed':True,'stages':{'usd':{'status':'success','outputs':{'object.usda':digest(self.rev/'object.usda')}}}})
    def test_review_is_explicit_and_tamper_invalidates(self):
        snap=candidate_snapshot(self.run,self.rev)
        self.assertFalse((self.rev/'review.json').exists())
        decide(self.run,self.rev,snap['fingerprint'],'accept')
        self.assertEqual(verified_selection(self.run)['selected'],'v_test')
        (self.rev/'object.usda').write_text('tampered')
        with self.assertRaises(ValueError): verified_selection(self.run)
        with self.assertRaises(ValueError): decide(self.run,self.rev,snap['fingerprint'],'accept')
    def test_reject_does_not_silently_fallback(self):
        snap=candidate_snapshot(self.run,self.rev);decide(self.run,self.rev,snap['fingerprint'],'reject')
        self.assertFalse((self.run/'usd/appearance/selection.json').exists())
        decide(self.run,self.rev,snap['fingerprint'],'original')
        self.assertEqual(verified_selection(self.run)['selected'],'original')
    def test_needs_alignment_cannot_be_accepted(self):
        atomic_json(self.rev/'report.json',{'status':'needs_alignment','stages':{}})
        snap=candidate_snapshot(self.run,self.rev)
        with self.assertRaises(ValueError): decide(self.run,self.rev,snap['fingerprint'],'accept')
    def test_reject_invalidates_previous_accept(self):
        snap=candidate_snapshot(self.run,self.rev);decide(self.run,self.rev,snap['fingerprint'],'accept')
        decide(self.run,self.rev,snap['fingerprint'],'reject')
        with self.assertRaises(ValueError): verified_selection(self.run)


class FeedbackTests(unittest.TestCase):
    def test_call_budget_reserves_failed_requests(self):
        import time
        report={'requests':[]};config={'max_completion_tokens':2048,'max_api_calls':1}
        self.assertIsNone(reserve(report,config,time.monotonic()))
        self.assertEqual(reserve(report,config,time.monotonic()),'api_call_budget')
        self.assertEqual(reserve({'requests':[]},{**config,'max_total_tokens':10},time.monotonic()),'token_budget')
        self.assertEqual(reserve({'requests':[]},{**config,'max_wall_seconds':1},time.monotonic()-2),'time_budget')
    def test_tools_are_typed_and_bounded(self):
        args={'elevation_degrees':5,'roll_degrees':0,'yaw_degrees':0,'scale_factor':1,'offset':[.5,.5],'reason':'align'}
        def message(name='adjust_camera'):return {'tool_calls':[{'id':'abc','function':{'name':name,'arguments':json.dumps(args)}}]}
        self.assertEqual(validate_action(message())[1],'adjust_camera')
        args['yaw_degrees']=181
        with self.assertRaises(ValueError): validate_action(message())
        with self.assertRaises(ValueError): validate_action(message('execute_python'))
        args['yaw_degrees']=float('nan')
        with self.assertRaises(ValueError): validate_action(message())

if __name__=='__main__':unittest.main()
