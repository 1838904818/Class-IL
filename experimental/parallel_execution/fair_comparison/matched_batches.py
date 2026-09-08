"""Prospective fixed-presentation sampler for a separately reviewed protocol.

Only already-arrived positive/negative index pools are supplied. No labels,
test scores or future-task arrays are consulted. Each pool is shuffled and
cycled independently; a row may repeat across a cycle boundary. Equal counts
do not establish equal unique-row coverage or full architectural fairness.
"""
import numpy as np


def batches(positive, negative, *, steps, positive_per_step, negative_per_step, seed):
    for name, value in (('steps',steps), ('positive_per_step',positive_per_step),
                        ('negative_per_step',negative_per_step)):
        if type(value) is not int or value < 1:
            raise ValueError(f'Invalid {name}')
    if type(seed) is not int or seed < 0:
        raise ValueError('Invalid seed')
    pools = []
    for source in (positive, negative):
        value = np.asarray(source)
        if value.ndim != 1 or value.dtype.kind not in 'iu' or len(value) == 0 or (value < 0).any():
            raise ValueError('Nonempty integer index pools required')
        if len(np.unique(value)) != len(value):
            raise ValueError('Duplicate index in source pool')
        pools.append(value)
    if np.intersect1d(*pools).size:
        raise ValueError('Positive and negative pools overlap')
    seeds = np.random.SeedSequence(seed).spawn(3)
    rngs = [np.random.default_rng(s) for s in seeds]
    orders = [rngs[i].permutation(pool) for i,pool in enumerate(pools)]
    offsets = [0,0]
    def take(i, size):
        parts = []
        while size:
            if offsets[i] == len(orders[i]):
                orders[i] = rngs[i].permutation(pools[i])
                offsets[i] = 0
            n = min(size, len(orders[i])-offsets[i])
            parts.append(orders[i][offsets[i]:offsets[i]+n])
            offsets[i] += n
            size -= n
        return np.concatenate(parts)
    for _ in range(steps):
        indices = np.concatenate([take(0, positive_per_step), take(1, negative_per_step)])
        labels = np.concatenate([np.ones(positive_per_step,np.int64), np.zeros(negative_per_step,np.int64)])
        order = rngs[2].permutation(len(indices))
        yield indices[order], labels[order]
