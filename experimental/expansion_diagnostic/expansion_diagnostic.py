"""Research-only counterfactual score decomposition, never a deployed scorer.

Float64 reference arithmetic is intentional. This is not a legacy fidelity
verifier, new attribution method, risk certificate or trained controller.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ScoreSnapshot:
    class_axis: tuple
    row_ids: tuple
    head: np.ndarray
    router_raw: np.ndarray

    def validate(self):
        if len(self.class_axis)<2 or len(set(self.class_axis))!=len(self.class_axis):
            raise ValueError('need unique class IDs and at least one rival')
        if not self.row_ids or len(set(self.row_ids))!=len(self.row_ids):
            raise ValueError('row IDs must be nonempty and unique')
        expected=(len(self.row_ids),len(self.class_axis))
        for v in [self.head,self.router_raw]:
            if np.asarray(v).shape!=expected or not np.isfinite(v).all():
                raise ValueError('nonfinite or mismatched score matrix')

def _joint(s,normalization_axis,weight,epsilon):
    cols=[s.class_axis.index(c) for c in normalization_axis]
    r=np.asarray(s.router_raw,dtype=np.float64)[:,cols]
    z=(r-r.mean(axis=1,keepdims=True))/(r.std(axis=1,keepdims=True)+epsilon)
    return np.asarray(s.head,dtype=np.float64)[:,cols]+weight*z

def _margin(scores,axis,target,rivals):
    t=axis.index(target); other=[axis.index(c) for c in rivals if c!=target]
    if not other: raise ValueError('target needs an observed rival')
    return scores[:,t]-scores[:,other].max(axis=1)

def decompose(old,new,target,weight=.5,epsilon=1e-8):
    """Return A/B/C/D margins and an order-specific exact score difference."""
    old.validate();new.validate()
    if old.row_ids!=new.row_ids:
        raise ValueError('identical ordered probe IDs required')
    if not set(old.class_axis).issubset(new.class_axis) or target not in old.class_axis:
        raise ValueError('target must be an old class and classes cannot disappear')
    if not np.isfinite(weight) or weight<0 or not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('invalid weight or epsilon')
    a=_margin(_joint(old,old.class_axis,weight,epsilon),old.class_axis,target,old.class_axis)
    b=_margin(_joint(new,old.class_axis,weight,epsilon),old.class_axis,target,old.class_axis)
    full=_joint(new,new.class_axis,weight,epsilon)
    c=_margin(full,new.class_axis,target,old.class_axis)
    d=_margin(full,new.class_axis,target,new.class_axis)
    effects={'state':b-a,'normalization_domain':c-b,'new_rival':d-c}
    residual=(d-a)-sum(effects.values())
    return {'targets':{'A':a,'B':b,'C':c,'D':d},'effects':effects,'total':d-a,'identity_residual':residual}
