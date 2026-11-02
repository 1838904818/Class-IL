"""Reassemble immutable NPZ bytes from published parts, verify SHA and load arrays."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.archive.resolve(strict=True);assert not a.output.exists();a.output.mkdir(parents=True)
    report=json.loads((root/'aggregate.json').read_text(encoding='utf-8'))
    for name,receipt in report['raw_arrays'].items():
        assert Path(name).name==name and name.endswith('.npz');dest=a.output/name;partial=dest.with_name(dest.name+'.partial');whole=hashlib.sha256();size=0
        with partial.open('xb') as out:
            for part in receipt['parts']:
                path=(root/part['path']).resolve(strict=True);assert path.is_relative_to(root);h=hashlib.sha256();count=0
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(1024**2),b''):out.write(block);h.update(block);whole.update(block);count+=len(block);size+=len(block)
                assert count==part['bytes'] and h.hexdigest()==part['sha256'],part['path']
        assert size==receipt['bytes'] and whole.hexdigest()==receipt['sha256'],name
        partial.replace(dest)
        with np.load(dest,allow_pickle=False) as arrays:
            assert set(arrays.files)==set(receipt['arrays'])
            for key,expected in receipt['arrays'].items():
                value=arrays[key];assert list(value.shape)==expected['shape'] and value.dtype.str==expected['dtype']
        print(json.dumps(dict(event='RAW_NPZ_RESTORED_SHA_AND_NUMPY_LOAD_VERIFIED',name=name,bytes=size,sha256=whole.hexdigest())),flush=True)
    print('ALL_SIX_RAW_NPZ_FILES_RESTORED_BYTE_EXACT_AND_ARRAYS_LOADED',flush=True)
if __name__=='__main__':main()
