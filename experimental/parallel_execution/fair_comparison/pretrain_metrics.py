"""Public aggregate metrics only; no SDK initialization or credential access."""
import math
import re


def step_metrics(measurement, *, attempt_id):
    if not isinstance(attempt_id, str) or not re.fullmatch('[a-f0-9]{32}', attempt_id):
        raise ValueError('Opaque attempt ID required')
    step = measurement['step']
    epoch = measurement['epoch']
    rows = measurement['raw_row_presentations']
    for value in (step, epoch, rows):
        if type(value) is not int or value < 0:
            raise ValueError('Invalid step, epoch or row count')
    if rows == 0:
        raise ValueError('Empty optimizer batch')
    result = {'pretrain/optimizer_step': step + 1, 'pretrain/epoch_index': epoch,
              'pretrain/batch_rows': rows}
    for source, target in [('loss','pretrain/cross_entropy'), ('batch_seconds','pretrain/batch_seconds'),
                           ('cuda_allocated_peak_bytes','resources/cuda_allocated_peak_bytes'),
                           ('cuda_reserved_peak_bytes','resources/cuda_reserved_peak_bytes')]:
        value = measurement.get(source)
        if value is None and source.startswith('cuda_'):
            continue
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid aggregate measurement')
        result[target] = value
    return {'event_id': f'{attempt_id}:{step}', 'metrics': result}


class AttemptEmitter:
    """Suppress duplicates in one live attempt only, not across process crashes.

    The caller supplies the transport. An uncertain send is not retried here.
    This class does not claim exactly-once remote delivery or persistent resume.
    """
    def __init__(self, attempt_id, send):
        self.attempt_id, self.send = attempt_id, send
        self.last_step = -1
        self.failed = False

    def emit(self, measurement):
        if self.failed:
            raise RuntimeError('Uncertain send requires external reconciliation')
        event = step_metrics(measurement, attempt_id=self.attempt_id)
        step = measurement['step']
        if step <= self.last_step:
            raise ValueError('Duplicate or out-of-order event')
        try:
            self.send(event)
        except Exception:
            self.failed = True
            raise
        self.last_step = step
