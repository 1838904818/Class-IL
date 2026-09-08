import unittest
import numpy as np
from matched_batches import batches


class MatchedTests(unittest.TestCase):
    def run_batches(self, positive, negative, seed=1):
        return list(batches(positive, negative, steps=5, positive_per_step=3, negative_per_step=4, seed=seed))

    def test_different_pools_equal_exposure(self):
        for positive,negative in (([0,1],[2,3]), (list(range(20)),list(range(20,60)))):
            result = self.run_batches(positive,negative)
            self.assertEqual(len(result),5)
            for indices,labels in result:
                self.assertEqual(len(indices),7)
                self.assertEqual(int(labels.sum()),3)
                self.assertTrue(set(indices[labels==1]).issubset(positive))
                self.assertTrue(set(indices[labels==0]).issubset(negative))

    def test_reproducible(self):
        a=self.run_batches([0,1],[2,3])
        b=self.run_batches([0,1],[2,3])
        for x,y in zip(a,b):
            for first,second in zip(x,y):
                np.testing.assert_array_equal(first,second)

    def test_reject_invalid_pools(self):
        for positive,negative in (([],[2]),([1,1],[2]),([1],[1]),([-1],[2]),([1.5],[2])):
            with self.assertRaises(ValueError):
                self.run_batches(positive,negative)


if __name__ == '__main__':
    unittest.main()
