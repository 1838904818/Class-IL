import copy
import tempfile
import unittest
import os
from pathlib import Path
import chunk_core as c


class Tests(unittest.TestCase):
    def targets(self):
        mapping={cid:dict(calibration=list(range(617)),roles={'trigger':dict(
            offsets=list(range(585,617)),ordinals=list(range(585,617)),
            row_ids=[c.digest([cid,i]) for i in range(32)])}) for cid in (0,1)}
        return c.all_targets(mapping)
    def record(self,item):
        return dict(status='EXTRACTION_RECORD_COMPLETE',science_sha256='a'*64,record_id=item['record_id'],checkpoint=item['checkpoint'],
            class_id=item['target']['class_id'],row_id=item['target']['row_id'],seed=item['seed'],
            repeat=item['repeat'],max_evals=157,batch_size=1,repeat_count=1,shap_version='0.51.0',
            masker_invariance='exact-value-equality',shap_attributions_computed=True,
            etg_actions_fitted=False,efficacy_metrics_computed=False,statistical_certificate=False,
            scientific_result_generated=False,archived_score_parity_verified=False,
            attributions=[0.]*78,native_calls=2,native_context_rows=210,
            base_value=0.,native_target=0.,additive_residual=0.)
    def test_complete_schedule(self):
        rows=c.schedule(self.targets())
        self.assertEqual(len(rows),1024);self.assertEqual(len({r['seed'] for r in rows}),8)
        self.assertEqual(len({r['record_id'] for r in rows}),1024)
        self.assertNotIn(20260911,{r['seed'] for r in rows})
        self.assertEqual(rows,c.schedule(self.targets()))
    def test_first_chunk_coverage(self):
        rows=c.schedule(self.targets())[:64]
        self.assertEqual({r['checkpoint'] for r in rows},{0,1})
        self.assertEqual({r['target']['class_id'] for r in rows},{0,1})
        self.assertEqual(len({r['target']['row_id'] for r in rows}),4)
    def test_tail_context_preserved(self):
        self.assertTrue(all(t['parent_rows']==105 for t in self.targets()))
    def test_write_read_resume(self):
        item=c.schedule(self.targets())[0];r=self.record(item)
        with tempfile.TemporaryDirectory() as d:
            entry=c.write_record(d,r,item,'a'*64)
            with self.assertRaises(ValueError):c.verified_resume(d,[entry],[item],'a'*64)
            entry=c.commit_record(d,entry,item,'a'*64)
            self.assertEqual(c.verified_resume(d,[entry],[item],'a'*64)[item['record_id']],r)
            with self.assertRaises(FileExistsError):c.write_record(d,r,item,'a'*64)
            with self.assertRaises(ValueError):c.verified_resume(d,[entry,entry],[item],'a'*64)
    def test_partial_file_is_not_reused(self):
        item=c.schedule(self.targets())[0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/(item['record_id']+'.json');path.write_bytes(b'{')
            with self.assertRaises(FileExistsError):c.write_record(d,self.record(item),item,'a'*64)
            with self.assertRaises(ValueError):c.verified_resume(d,[dict(record_id=item['record_id'],bytes=1,sha256='0'*64)],[item],'a'*64)
    def test_wrong_seed_and_science_rejected(self):
        item=c.schedule(self.targets())[0];r=self.record(item)
        for key in ('seed','science_sha256','checkpoint','row_id','max_evals','repeat_count'):
            value=copy.deepcopy(r);value[key]='tampered'
            with self.assertRaises(ValueError):c.check_record(value,item,'a'*64)
    def test_nonfinite_rejected(self):
        item=c.schedule(self.targets())[0];r=self.record(item);r['attributions'][0]=float('nan')
        with self.assertRaises(ValueError):c.check_record(r,item,'a'*64)
    def test_unreviewed_file_blocks_resume(self):
        item=c.schedule(self.targets())[0]
        with tempfile.TemporaryDirectory() as d:
            entry=c.commit_record(d,c.write_record(d,self.record(item),item,'a'*64),item,'a'*64)
            (Path(d)/'unexpected.json').write_bytes(b'{}')
            with self.assertRaises(ValueError):c.verified_resume(d,[entry],[item],'a'*64)
    def test_tampered_commit_blocks_resume(self):
        item=c.schedule(self.targets())[0]
        with tempfile.TemporaryDirectory() as d:
            entry=c.commit_record(d,c.write_record(d,self.record(item),item,'a'*64),item,'a'*64)
            (Path(d)/(item['record_id']+'.commit.json')).write_bytes(b'{}')
            with self.assertRaises(ValueError):c.verified_resume(d,[entry],[item],'a'*64)
    def test_same_seed_set_for_every_pair(self):
        rows=c.schedule(self.targets())
        for target in self.targets():
            for cp in (0,1):
                self.assertEqual([r['seed'] for r in rows if r['target']==target and r['checkpoint']==cp],list(range(20270000,20270008)))
    def test_no_etg_on_incomplete_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):c.completion_gate(d,[],c.schedule(self.targets()),'a'*64)
    def test_context_count_rejects_float(self):
        item=c.schedule(self.targets())[0];r=self.record(item);r['native_context_rows']=210.0
        with self.assertRaises(ValueError):c.check_record(r,item,'a'*64)
    def test_schedule_does_not_alias_inputs(self):
        targets=self.targets();items=c.schedule(targets);before=copy.deepcopy(items)
        targets[0]['source_offset']+=1
        self.assertEqual(items,before)
        items[0]['target']['source_offset']+=1
        self.assertEqual(items[1],before[1])
    def test_item_mutation_rejected(self):
        item=c.schedule(self.targets())[0];r=self.record(item)
        item['target']['source_offset']+=1
        with self.assertRaises(ValueError):c.check_record(r,item,'a'*64)
    def test_nonpaired_seed_rejected_even_with_new_digest(self):
        item=c.schedule(self.targets())[0];item['seed']=123
        item['record_id']=c.digest({k:v for k,v in item.items() if k!='record_id'})
        with self.assertRaises(ValueError):c.check_record(self.record(item),item,'a'*64)
    def test_complete_records_and_missing_record(self):
        items=c.schedule(self.targets())
        with tempfile.TemporaryDirectory() as d:
            entries=[c.commit_record(d,c.write_record(d,self.record(i),i,'a'*64),i,'a'*64) for i in items]
            self.assertEqual(len(c.completion_gate(d,entries,items,'a'*64)),1024)
            with self.assertRaises(ValueError):c.completion_gate(d,entries[:-1],items,'a'*64)
    def test_write_rejects_escape_id(self):
        i=c.schedule(self.targets())[0];i['record_id']='../escape'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):c.write_record(d,self.record(i),i,'a'*64)
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_wrong_entry_id(self):
        i=c.schedule(self.targets())[0]
        with tempfile.TemporaryDirectory() as d:
            entry=c.write_record(d,self.record(i),i,'a'*64);entry['record_id']='f'*64
            with self.assertRaises(ValueError):c.commit_record(d,entry,i,'a'*64)
    def test_link_directory_rejected_before_write(self):
        i=c.schedule(self.targets())[0]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'real').mkdir()
            try:os.symlink(root/'real',root/'link',target_is_directory=True)
            except OSError as error:self.skipTest('OS does not permit symlink: '+str(error))
            with self.assertRaises(ValueError):c.write_record(root/'link',self.record(i),i,'a'*64)
            self.assertEqual(list((root/'real').iterdir()),[])

if __name__=='__main__':unittest.main()
