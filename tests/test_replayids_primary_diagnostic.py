"""Numerical rejection tests plus full packaged-result recomputation."""
import copy
import json
from pathlib import Path
import statistics as st
import sys
import unittest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools'))
from replayids_primary_diagnostic import audit_cps, analyze

def metrics(cm,tasks):
    n=len(cm);rows=[];total=sum(map(sum,cm))
    for i in range(n):
        tp=cm[i][i];support=sum(cm[i]);pred=sum(r[i] for r in cm)
        p=tp/pred if pred else 0;r=tp/support if support else 0
        rows.append(dict(class_id=i,class_name=str(i),support=support,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0))
    return dict(confusion_matrix=cm,total_rows=total,accuracy=sum(cm[i][i] for i in range(n))/total,
        per_class=rows,macro_f1=st.mean(r['f1'] for r in rows),balanced_accuracy=st.mean(r['recall'] for r in rows),
        binary_detection=dict(benign_false_positive_rate=1-cm[0][0]/sum(cm[0]),attack_detection_recall=1-sum(row[0] for row in cm[1:])/(total-sum(cm[0]))),
        task_accuracy={str(t):sum(cm[i][i] for i in task)/sum(sum(cm[i]) for i in task) for t,task in enumerate(tasks) if max(task)<n})

def fixture():
    tasks=[[0,1],[2]]; ms=[metrics([[8,2],[1,9]],tasks),metrics([[7,2,1],[1,7,2],[0,2,8]],tasks)]
    cps=[dict(checkpoint=i,seen_classes=list(range(len(m['confusion_matrix']))),metrics=m) for i,m in enumerate(ms)]
    matrix=[[ms[0]['task_accuracy']['0'],None],[ms[1]['task_accuracy']['0'],ms[1]['task_accuracy']['1']]]
    s=dict(task_accuracy_matrix=matrix,average_task_accuracy=st.mean(matrix[-1]),average_forgetting=matrix[0][0]-matrix[1][0],
        final_overall_accuracy=ms[1]['accuracy'],final_macro_f1=ms[1]['macro_f1'],final_balanced_accuracy=ms[1]['balanced_accuracy'],
        final_attack_detection_recall=ms[1]['binary_detection']['attack_detection_recall'],final_benign_false_positive_rate=ms[1]['binary_detection']['benign_false_positive_rate'])
    return cps,s,tasks

class Tests(unittest.TestCase):
    def test_metrics_validate(self): audit_cps(*fixture())
    def test_binary_and_confusion_corruption_rejected(self):
        cps,s,t=fixture();cps[-1]['metrics']['binary_detection']['attack_detection_recall']=.999
        with self.assertRaises(ValueError): audit_cps(cps,s,t)
        cps,s,t=fixture();cps[-1]['metrics']['confusion_matrix'][0][0]=True
        with self.assertRaises(ValueError): audit_cps(cps,s,t)
    def test_missing_checkpoint_rejected(self):
        cps,s,t=fixture()
        with self.assertRaises(ValueError): audit_cps(cps[:-1],s,t)
    def test_exact_release_recomputation(self):
        path=REPO/'results/replayids-primary-diagnostic/DIAGNOSTIC.json'
        expected=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(analyze(REPO),expected)
        self.assertIn('last epoch',expected['checkpoint_policies']['primary'])
        self.assertNotEqual(expected['cohorts']['all_five_descriptive']['versus_replay']['primary'],
                            expected['cohorts']['all_five_descriptive']['versus_replay']['guard'])

if __name__=='__main__': unittest.main()
