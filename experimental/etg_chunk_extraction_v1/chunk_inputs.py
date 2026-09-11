"""One bound trigger's original context and fixed Task-0 reference pool.

Tensor materialization is compute work, never a login-node preflight.
"""
import numpy as np
from chunk_core import all_targets, require, validate_item


class RecordInputs:
    def __init__(self, mapping, source_root, native_root, item, opener=None):
        validate_item(item)
        if opener is None:
            from bundle import open_features
            opener=open_features
        planned=all_targets(mapping)
        target=item['target']
        require(target in planned,'target differs from reconstructed bound metadata')
        self.target=target.copy()
        self.mapping=mapping
        self.source_root=source_root
        self.native_root=native_root
        refs=[]
        for cid in (0,1):
            rec=mapping[cid]
            source=opener(source_root,rec['source'],rec['source_rows'])
            reference=np.array(source[rec['background']],copy=True,order='C')
            require(reference.dtype==np.float32 and reference.shape==(16,78)
                    and np.isfinite(reference).all(),'reference contract')
            refs.append(reference)
            if cid==target['class_id']:
                cal=opener(native_root,rec['calibration_file'],len(rec['calibration']))
                start,n=target['parent_start'],target['parent_rows']
                self.parent=np.array(cal[start:start+n],copy=True,order='C')
                offsets=np.asarray(rec['calibration'])[start:start+n]
                require(self.parent.dtype==np.float32 and self.parent.shape==(n,78)
                    and np.isfinite(self.parent).all()
                    and np.array_equal(self.parent,source[offsets]),'native parent/source mismatch')
                require(int(offsets[target['row_in_parent']])==target['source_offset'],
                        'target source offset mismatch')
                del cal
            del source
        self.references=np.concatenate(refs,axis=0)
        require(self.references.shape==(32,78),'fixed reference count')
        self.parent.flags.writeable=False
        self.references.flags.writeable=False

    def recheck_files(self):
        from native_loader import safe_file, sha
        for cid in (0,1):
            rec=self.mapping[cid]
            refs=[(self.source_root,rec['source'])]
            if cid==self.target['class_id']:
                refs.append((self.native_root,rec['calibration_file']))
            for root,ref in refs:
                require(sha(safe_file(root,ref['path']))==ref['sha256'],'input changed during extraction')
