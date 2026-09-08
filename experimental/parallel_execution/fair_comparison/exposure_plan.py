"""Count the existing L1 batch schedule without training or inspecting test data."""


def plan(task_counts, *, epochs, batch_size, negative_ratio, exemplar_capacity):
    for value in (epochs, batch_size, negative_ratio, exemplar_capacity):
        if type(value) is not int or value < 1:
            raise ValueError('Positive integer configuration required')
    width = max(1, batch_size // (negative_ratio + 1))
    prior = 0
    seen = set()
    records = []
    if not task_counts:
        raise ValueError('Empty stream')
    for task, counts in enumerate(task_counts):
        if not counts:
            raise ValueError('Empty task')
        for c, n in counts.items():
            if type(c) is not int or c < 0 or c in seen or type(n) is not int or n < 1:
                raise ValueError('Invalid arriving class or count')
            seen.add(c)
        current = sum(counts.values())
        for c, n in counts.items():
            negative = min(current - n + prior, n * negative_ratio)
            if not negative:
                raise ValueError('Empty negative pool')
            records.append(dict(task=task, class_id=c, positive_presentations=n * epochs,
                                negative_presentations=negative * epochs,
                                row_presentations=(n + negative) * epochs,
                                optimizer_steps=((n + width - 1) // width) * epochs))
        prior += sum(min(exemplar_capacity, n) for n in counts.values())
    return dict(scope='planned_per_loss_arm_not_executed', classes=records,
                optimizer_steps=sum(x['optimizer_steps'] for x in records),
                row_presentations=sum(x['row_presentations'] for x in records))
