import unittest
import numpy as np
import exposure_plan as p
import l1_runner as r


class ExposureTests(unittest.TestCase):
    def test_real_counts(self):
        for normal, steps, rows in ((5559, 19720, 1836910), (124780, 35400, 3195890)):
            result = p.plan([{0: normal, 1: 5559}, {2:124780, 3:2970},
                             {4:3130, 5:4287}, {6:6, 7:3185}], epochs=10,
                            batch_size=384, negative_ratio=4, exemplar_capacity=50)
            self.assertEqual((result['optimizer_steps'], result['row_presentations']), (steps, rows))

    def test_matches_actual_generator(self):
        counts = [{0: 7, 1: 3}, {2: 13, 3: 2}]
        tasks = [list(c) for c in counts]
        labels = np.array([c for group in counts for c,n in group.items() for _ in range(n)])
        available = np.array([t for t,group in enumerate(counts) for n in group.values() for _ in range(n)])
        train = dict(labels=labels, available_tasks=available)
        for batch_size in (3, 8, 16):
            config = dict(seed=1, epochs=2, batch_size=batch_size, negative_ratio=4, exemplar_capacity=2)
            predicted = p.plan(counts, **{k:v for k,v in config.items() if k != 'seed'})
            steps = rows = 0
            for t, group in enumerate(counts):
                buffers = r.buffers_for(train, tasks[:t], config)
                for c in group:
                    for epoch in range(2):
                        for indices, _, _ in r.batches_for(train, tasks, buffers, config, t, c, epoch):
                            steps += 1
                            rows += len(indices)
            self.assertEqual((steps, rows), (predicted['optimizer_steps'], predicted['row_presentations']))

    def test_reject_duplicate_class(self):
        with self.assertRaises(ValueError):
            p.plan([{0:3,1:3},{0:2}],epochs=1,batch_size=8,negative_ratio=4,exemplar_capacity=2)


if __name__ == '__main__':
    unittest.main()
