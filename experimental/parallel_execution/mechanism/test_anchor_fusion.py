import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from anchor_fusion import check_transition, execute, fuse, no_links, sha, strict_json, winner


class FusionTests(unittest.TestCase):
    def example(self, raw_new=-100., head_new=0.):
        return check_transition([[.2,.8]], [[0.,-1.]], [[.2,.8,head_new]], [[0.,-1.,raw_new]],
             ['a','b'], ['a','b','c'], ['a','b'], weight=.5,floor=1e-8,fixed_scale=1.,tolerance=1e-12)

    def test_irrelevant_new_class_changes_old_winner_in_all_seen(self):
        r=self.example()
        self.assertEqual(r['controls']['all_seen']['old_restricted_winner_changes'],1)
        self.assertEqual(r['controls']['all_seen']['new_class_winners'],0)
        self.assertEqual(r['controls']['reference']['old_restricted_winner_changes'],0)
        self.assertTrue(r['controls']['reference']['conditional_invariance_pass'])

    def test_new_rival_can_still_win(self):
        r=self.example(2.,1.)
        self.assertEqual(r['controls']['reference']['new_class_winners'],1)
        self.assertEqual(r['controls']['reference']['old_restricted_winner_changes'],0)

    def test_many_random_frozen_expansions(self):
        rng=np.random.default_rng(8)
        for k in (3,5,17):
            oldh=rng.uniform(size=(19,3));oldr=-rng.uniform(size=(19,3))
            h=np.column_stack((oldh,rng.uniform(size=(19,k))));r=np.column_stack((oldr,-rng.uniform(size=(19,k))))
            out=check_transition(oldh,oldr,h,r,['a','b','c'],['a','b','c']+[f'n{i}' for i in range(k)],
                 ['a','b'],weight=.5,floor=.01,fixed_scale=.5,tolerance=1e-12)
            self.assertTrue(out['controls']['reference']['conditional_invariance_pass'])

    def test_permutation_and_tie_policy(self):
        self.assertEqual(winner([[1,1]],['z','a'])[0],'a')
        args=dict(weight=.5,floor=.01)
        a=fuse([[.2,.8,.5]],[[0,-1,-100]],['a','b','c'],['a','b'],**args)
        b=fuse([[.5,.8,.2]],[[-100,-1,0]],['c','b','a'],['b','a'],**args)
        np.testing.assert_array_equal(a,b[:,[2,1,0]])

    def test_batch_independence(self):
        h=np.array([[.2,.8],[.4,.1]]);r=np.array([[0,-1],[3,7]])
        kwargs=dict(classes=['a','b'],reference_classes=['a','b'],weight=.5,floor=.01)
        np.testing.assert_array_equal(fuse(h,r,**kwargs),np.concatenate([fuse(h[i:i+1],r[i:i+1],**kwargs) for i in range(2)]))

    def test_scale_degenerate_is_finite(self):
        np.testing.assert_array_equal(fuse([[.2,.8]],[[1,1]],['a','b'],['a','b'],weight=.5,floor=.01),[[.2,.8]])

    def test_legacy_formula_is_separate_control(self):
        h=np.array([[.2,.8]]);r=np.array([[0.,-1.]])
        actual=fuse(h,r,['a','b'],['a','b'],weight=.5,floor=.01,mode='all_seen_legacy_formula')
        expected=h+.5*(r-r.mean(1,keepdims=True))/(r.std(1,keepdims=True)+.01)
        np.testing.assert_array_equal(actual,expected)

    def test_overflowing_statistics_rejected(self):
        with np.errstate(over='ignore',invalid='ignore'):
            with self.assertRaises(ValueError):
                fuse([[.2,.8]],[[1e308,-1e308]],['a','b'],['a','b'],weight=.5,floor=.01)

    def test_changed_old_state_is_not_invariance_evidence(self):
        r=check_transition([[.2,.8]],[[0,-1]],[[.3,.8,0]],[[0,-1,-100]],
            ['a','b'],['a','b','c'],['a','b'],weight=.5,floor=.01,fixed_scale=1,tolerance=1e-12)
        self.assertFalse(r['frozen_old_input_scores'])
        self.assertIsNone(r['controls']['reference']['conditional_invariance_pass'])

    def test_invalid_parameters_and_inputs(self):
        for w,f in [(float('nan'),.1),(-1,.1),(.5,0),(.5,float('inf'))]:
            with self.assertRaises(ValueError):fuse([[.2,.8]],[[0,1]],['a','b'],['a','b'],weight=w,floor=f)
        with self.assertRaises(ValueError):fuse([[.2,1.8]],[[0,1]],['a','b'],['a','b'],weight=.5,floor=.1)
        with self.assertRaises(ValueError):fuse([[.2,.8]],[[0,1]],['a','a'],['a','b'],weight=.5,floor=.1)
        with self.assertRaises(ValueError):fuse([[.2,.8]],[[0,1]],['a','b'],['a','c'],weight=.5,floor=.1)

    def fixture(self, root):
        arrays={'old_head':np.array([[.2,.8]]),'new_head':np.array([[.2,.8,0.]]),
            'old_raw':np.array([[0.,-1.]]),'new_raw':np.array([[0.,-1.,-100.]]),
            'old_row_ids':np.array(['row1']),'new_row_ids':np.array(['row1'])}
        bindings={}
        for n,a in arrays.items():
            np.save(root/(n+'.npy'),a,allow_pickle=False)
            bindings[n]={'path':n+'.npy','sha256':sha(root/(n+'.npy'))}
        p={'schema':'expansion-fusion-pilot-v1','status':'BOUND','evidence_kind':'synthetic','independent_binary_head_probabilities':True,
           'reference_origin':'task0_train_only_frozen_router_bank','reference_classes':['a','b'],
           'old_classes':['a','b'],'new_classes':['a','b','c'],'arrays':bindings,
           'weight':.5,'floor':.01,'fixed_scale':1.,'tolerance':1e-12,'block_rows':1}
        for key in ('reference_state_sha256','encoder_state_sha256','feature_contract_sha256','score_export_receipt_sha256'):p[key]='a'*64
        manifest=root/'manifest.json';manifest.write_text(json.dumps(p),encoding='utf-8')
        return manifest,p

    def test_full_file_pipeline_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';root.mkdir();m,p=self.fixture(root);out=Path(d)/'out'
            r=execute(m,out)
            self.assertEqual(r['status'],'SCORE_PILOT_COMPLETE')
            self.assertFalse(r['classification_performance_measured'])
            self.assertTrue(r['conditional_invariance_pass'])
            self.assertEqual(json.loads((out/'COMPLETE.json').read_text())['result_sha256'],sha(out/'RESULT.json'))
            with self.assertRaises(ValueError):execute(m,out)

    def test_tamper_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';root.mkdir();m,p=self.fixture(root)
            np.save(root/'old_head.npy',np.array([[.3,.8]]))
            with self.assertRaises(ValueError):execute(m,Path(d)/'out')
            self.assertFalse((Path(d)/'out').exists())

    def test_alignment_and_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';root.mkdir();m,p=self.fixture(root)
            p['arrays']['new_head']['path']='../input/new_head.npy'
            m.write_text(json.dumps(p))
            with self.assertRaises(ValueError):execute(m,Path(d)/'out')

    def test_strict_nonfinite_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'
            for s in ('{"a":1,"a":2}','{"a":1e400}','{"a":NaN}'):
                p.write_text(s)
                with self.assertRaises(ValueError):strict_json(p)

    def test_numeric_strings_are_not_score_arrays(self):
        with self.assertRaises(ValueError):
            fuse([['.2','.8']], [[0.,-1.]], ['a','b'], ['a','b'], weight=.5, floor=.01)
        with self.assertRaises(ValueError):
            fuse([[.2,.8]], [['0','-1']], ['a','b'], ['a','b'], weight=.5, floor=.01)

    def test_windows_reparse_flag_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'exists', return_value=True), patch.object(Path, 'lstat', return_value=SimpleNamespace(st_file_attributes=1024)):
                with self.assertRaises(ValueError): no_links(Path(d)/'payload.json')


if __name__=='__main__':unittest.main()
