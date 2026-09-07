import unittest
import numpy as np
from expansion_diagnostic import ScoreSnapshot,decompose

def snap(axis,h,r,rows=('probe-a',)):
    return ScoreSnapshot(tuple(axis),rows,np.array(h,dtype=float),np.array(r,dtype=float))

class Tests(unittest.TestCase):
    def test_identical_no_effect(self):
        s=snap([2,7],[[.8,.2]],[[-1,-2]])
        d=decompose(s,s,2)
        for x in d['effects'].values():np.testing.assert_allclose(x,0,atol=1e-14)
    def test_expansion_without_old_state_change(self):
        a=snap([2,7],[[.8,.2]],[[-1,-2]])
        b=snap([2,7,99],[[.8,.2,.95]],[[-1,-2,-.1]])
        d=decompose(a,b,2)
        np.testing.assert_allclose(d['effects']['state'],0)
        self.assertNotEqual(float(d['effects']['normalization_domain'][0]),0)
        self.assertLess(float(d['effects']['new_rival'][0]),0)
        np.testing.assert_allclose(d['identity_residual'],0,atol=1e-14)
    def test_new_rival_effect_never_positive(self):
        rng=np.random.default_rng(42)
        a=snap([2,7],rng.random((20,2)),rng.normal(size=(20,2)),tuple(range(20)))
        b=snap([2,7,99],rng.random((20,3)),rng.normal(size=(20,3)),tuple(range(20)))
        d=decompose(a,b,2)
        self.assertTrue((d['effects']['new_rival']<=1e-14).all())
        np.testing.assert_allclose(d['identity_residual'],0,atol=1e-14)
    def test_class_axis_permutation(self):
        a=snap([2,7],[[.8,.2]],[[-1,-2]])
        b=snap([99,7,2],[[.95,.2,.8]],[[-.1,-2,-1]])
        c=snap([2,7,99],[[.8,.2,.95]],[[-1,-2,-.1]])
        for k in ['state','normalization_domain','new_rival']:
            np.testing.assert_allclose(decompose(a,b,2)['effects'][k],decompose(a,c,2)['effects'][k])
    def test_state_only_change(self):
        a=snap([2,7],[[.8,.2]],[[-1,-2]])
        b=snap([2,7],[[.6,.3]],[[-1,-2]])
        d=decompose(a,b,2)
        np.testing.assert_allclose(d['effects']['state'],-.3)
        np.testing.assert_allclose(d['effects']['new_rival'],0)
        np.testing.assert_allclose(d['effects']['normalization_domain'],0)
    def test_misaligned_rows_rejected(self):
        a=snap([2,7],[[.8,.2]],[[-1,-2]])
        b=snap([2,7],[[.8,.2]],[[-1,-2]],('other',))
        with self.assertRaises(ValueError):decompose(a,b,2)
    def test_unseen_target_rejected(self):
        a=snap([2,7],[[.8,.2]],[[-1,-2]])
        with self.assertRaises(ValueError):decompose(a,a,99)
    def test_nonfinite_rejected(self):
        a=snap([2,7],[[.8,np.nan]],[[-1,-2]])
        with self.assertRaises(ValueError):decompose(a,a,2)
    def test_degenerate_distances_finite(self):
        a=snap([2,7],[[.8,.2]],[[-1,-1]])
        np.testing.assert_allclose(decompose(a,a,2)['total'],0)
    def test_duplicate_class_rejected(self):
        a=snap([2,2],[[.8,.2]],[[-1,-2]])
        with self.assertRaises(ValueError):decompose(a,a,2)

if __name__=='__main__':unittest.main()
