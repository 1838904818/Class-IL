import copy
import unittest
import numpy as np
from chunk_core import all_targets, schedule, digest
from chunk_inputs import RecordInputs


class InputsTests(unittest.TestCase):
    def setUp(self):
        self.mapping={};self.arrays={}
        for cid in (0,1):
            source=np.arange(700*78,dtype=np.float32).reshape(700,78)+cid
            cal=np.arange(617,dtype=np.int64)
            ordinals=[0]+list(range(586,617))
            self.mapping[cid]=dict(calibration=cal,source_rows=700,source={'path':f's{cid}'},
                calibration_file={'path':f'c{cid}'},background=list(range(620,636)),
                roles={'trigger':dict(offsets=ordinals,ordinals=ordinals,
                    row_ids=[digest([cid,i]) for i in ordinals])})
            self.arrays[f's{cid}']=source
            self.arrays[f'c{cid}']=source[cal].copy()
        self.items=schedule(all_targets(self.mapping))
    def open(self,root,ref,rows):
        a=self.arrays[ref['path']]
        self.assertEqual(a.shape,(rows,78));return a
    def test_full_and_tail_are_original_contexts(self):
        for index in (0,16,32,48):
            item=self.items[index]
            data=RecordInputs(self.mapping,'s','n',item,self.open)
            t=item['target'];a=self.arrays[f"c{t['class_id']}"]
            np.testing.assert_array_equal(data.parent,a[t['parent_start']:t['parent_start']+t['parent_rows']])
            self.assertEqual(data.parent.shape[0],512 if index<32 else 105)
            self.assertFalse(data.parent.flags.writeable)
    def test_references_identical_across_checkpoints_and_classes(self):
        refs=[RecordInputs(self.mapping,'s','n',self.items[i],self.open).references for i in (0,8,16,24,32,40,48,56)]
        for a in refs:np.testing.assert_array_equal(a,refs[0])
    def test_mismatched_parent_is_rejected(self):
        self.arrays['c0'][5,0]+=1
        with self.assertRaises(ValueError):RecordInputs(self.mapping,'s','n',self.items[0],self.open)
    def test_metadata_mismatch_even_with_rehashed_item(self):
        item=copy.deepcopy(self.items[0]);item['target']['source_offset']=20
        item['record_id']=digest({k:v for k,v in item.items() if k!='record_id'})
        with self.assertRaises(ValueError):RecordInputs(self.mapping,'s','n',item,self.open)

if __name__=='__main__':unittest.main()
