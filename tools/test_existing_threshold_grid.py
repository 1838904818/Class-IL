import copy
import itertools
import unittest
from audit_existing_threshold_grid import verify_grid

def fixture():
    a={'transition_rows':[{'delta_recall':-.05,**{f'jaccard_top{k}':.6 for k in [5,10,15,20]}}], 'threshold_sensitivity':[]}
    for k,j,d in itertools.product([5,10,15,20],[.5,.6,.7,.8],[0,.02,.05,.1]):
        n=1 if d==.1 else 0
        e=1 if n and j in [.7,.8] else 0
        a['threshold_sensitivity'].append({'k':k,'jaccard_threshold':j,'allowed_recall_drop':d,'events':e,'eligible_transitions':n,'rate':e/n if n else None})
    return a

class Tests(unittest.TestCase):
    def test_exact_recall_boundary_excluded(self):
        rows=verify_grid(fixture())
        self.assertTrue(all(r['eligible_transitions']==0 for r in rows if r['allowed_recall_drop']==.05))
    def test_exact_jaccard_boundary_excluded(self):
        rows=verify_grid(fixture())
        self.assertTrue(all(r['events']==0 for r in rows if r['jaccard_threshold']==.6))
    def test_tampered_event_rejected(self):
        a=fixture();a['threshold_sensitivity'][0]['events']=1
        with self.assertRaises(ValueError):verify_grid(a)
    def test_duplicate_setting_rejected(self):
        a=fixture();a['threshold_sensitivity'][1]=copy.deepcopy(a['threshold_sensitivity'][0])
        with self.assertRaises(ValueError):verify_grid(a)
    def test_null_denominator_enforced(self):
        a=fixture();a['threshold_sensitivity'][0]['rate']=0
        with self.assertRaises(ValueError):verify_grid(a)

if __name__=='__main__':unittest.main()
