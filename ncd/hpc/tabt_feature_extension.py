"""TRAIN-only CSE/NSL column typing for the existing ER TabTransformer.

Low cardinality is an explicit extension engineering default (32 values), not
an attributed paper hyperparameter. Numeric inputs remain unchanged. CSE's
already TRAIN-fitted scaler is inverted only to identify integer-valued columns.
No TEST values, calibration rows, or future-task rows fit the vocabularies.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from ncd.section81_tabtransformer import TabularClassifier as OriginalClassifier


def fit_feature_spec(dataset,feature_names,task0_training_x):
    if dataset not in ('NSL-KDD','CSE-CIC-IDS2018'):raise ValueError(dataset)
    names=list(feature_names);x=np.asarray(task0_training_x,dtype=np.float32)
    assert x.ndim==2 and x.shape[1]==len(names) and len(x) and len(set(names))==len(names)
    assert np.isfinite(x).all()
    slug='nsl-kdd' if dataset=='NSL-KDD' else 'cic-ids-2018'
    metadata=json.loads((Path(os.environ['NCD_DATA_ROOT'])/slug/'data_quality.json').read_text())
    assert metadata['feature_names']==names
    scaler=metadata.get('scaler')
    groups=[];decisions=[]
    for column,name in enumerate(names):
        values=np.unique(x[:,column])
        raw=values.astype(np.float64)
        if scaler is not None:raw=raw*scaler['scale'][column]+scaler['mean'][column]
        integer=bool(np.allclose(raw,np.rint(raw),rtol=0,atol=max(1e-5,float(np.max(np.abs(raw)))*5e-7)))
        categorical=bool(1<len(values)<=32 and integer)
        if categorical:groups.append(dict(name=name,columns=[column],kind='numeric',values=values.tolist()))
        decisions.append(dict(name=name,task0_unique_values=len(values),integer_valued=integer,
                              path='categorical' if categorical else 'continuous'))
    categorical={i for g in groups for i in g['columns']}
    return dict(dataset=dataset,feature_names=names,groups=groups,
        continuous_columns=[i for i in range(len(names)) if i not in categorical],
        fit_scope='Task-0 optimization-eligible TRAIN only',unknown_id=0,
        fit_rows=len(x),test_used_for_fitting=False,decisions=decisions,
        extension_scope='Unregistered dataset extension; unchanged ER training and model architecture',
        categorical_rule='2..32 unique integer values; engineering default, no dataset tuning',
        integer_check='Original units via frozen TRAIN scaler where available; otherwise prepared values')


class TabularClassifier(OriginalClassifier):
    def encode(self,x):
        if self.spec['groups']:return super().encode(x)
        return torch.empty((len(x),0),dtype=torch.long,device=x.device),x.index_select(1,self.continuous_columns)
