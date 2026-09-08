import unittest
import pretrain_metrics as m


class MetricsTests(unittest.TestCase):
    def row(self):
        return dict(step=0, epoch=0, raw_row_presentations=384, loss=0.7,
                    batch_seconds=1.0, cuda_allocated_peak_bytes=100,
                    cuda_reserved_peak_bytes=200, secret='must-not-leave',
                    row_ids=['private'], path='private', storage={'private':1})

    def test_allowlist(self):
        event=m.step_metrics(self.row(),attempt_id='a'*32)
        self.assertEqual(len(event['metrics']),7)
        self.assertNotIn('private',str(event))
        self.assertNotIn('secret',str(event))
        self.assertEqual(event['metrics']['pretrain/optimizer_step'],1)

    def test_invalid_numeric(self):
        for key,value in [('loss',float('nan')),('batch_seconds',-1),('step',True),('raw_row_presentations',0)]:
            row=self.row();row[key]=value
            with self.assertRaises(ValueError):m.step_metrics(row,attempt_id='a'*32)

    def test_duplicate(self):
        sent=[]; emitter=m.AttemptEmitter('a'*32,sent.append)
        emitter.emit(self.row())
        with self.assertRaises(ValueError):emitter.emit(self.row())
        self.assertEqual(len(sent),1)

    def test_uncertain_no_retry(self):
        calls=[]
        def fail(event):
            calls.append(event)
            raise OSError('transport')
        emitter=m.AttemptEmitter('a'*32,fail)
        with self.assertRaises(OSError):emitter.emit(self.row())
        with self.assertRaises(RuntimeError):emitter.emit(self.row())
        self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
