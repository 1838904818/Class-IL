"""Single-attempt aggregate transport with explicit uncertain-delivery records.

SDK injection permits network-free tests. Existing output directories fail
closed: recovering an interrupted sender requires separate reconciliation,
never automatic retraining or blind resending. SDK acceptance is not proof of
remote persistence. The caller verifies committed checkpoint provenance.
"""
from pathlib import Path
import copy
import checkpoint_metrics as metrics
import tracking_policy as policy_module
import l1_runner as r


class Tracker:
    def __init__(self, directory, policy, run_id, sdk):
        arguments = policy_module.init_arguments(policy, run_id=run_id, resume=False)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.sdk = sdk
        self.closed = False
        self.state = {'status': 'initialization_uncertain', 'run_id': run_id,
                      'entity': arguments['entity'], 'project': arguments['project'],
                      'events': [], 'remote_verified': False}
        self.save()
        settings = sdk.Settings(console='off', disable_code=True, disable_git=True,
            disable_job_creation=True, save_code=False, x_disable_meta=True,
            x_disable_stats=True, x_disable_machine_info=True,
            x_save_requirements=False, x_stats_track_process_tree=False, silent=True)
        self.run = sdk.init(**arguments, settings=settings, dir=str(self.directory))
        if str(self.run.id) != run_id or str(self.run.entity) != arguments['entity'] or str(self.run.project) != arguments['project']:
            self.state['status'] = 'destination_mismatch'
            self.save()
            self.run.finish(exit_code=1)
            raise ValueError('SDK destination differs from approved destination')
        self.state['status'] = 'ready'
        self.save()

    def save(self):
        r.atomic_json(self.directory / 'DELIVERY.json', self.state)

    def send(self, checkpoint, *, arm, checkpoint_sha256, history):
        if self.closed or self.state['status'] != 'ready':
            raise RuntimeError('Sender is closed or requires reconciliation')
        event = metrics.checkpoint_payload(checkpoint, arm=arm,
            checkpoint_sha256=checkpoint_sha256, history=history)
        if any(x['event_id'] == event['event_id'] for x in self.state['events']):
            raise ValueError('Duplicate event')
        # Persist intent before invoking any SDK upload method. A crash leaves
        # this event uncertain, even if run.log returned or remotely succeeded.
        item = {'event_id': event['event_id'], 'status': 'send_uncertain'}
        self.state['events'].append(item)
        self.state['status'] = 'send_uncertain'
        self.save()
        payload = dict(event['metrics'], event_id=event['event_id'])
        for name in ('per_class', 'confusion'):
            table = event[name]
            payload[f'{arm}/{name}_task_{checkpoint["task"]}'] = self.sdk.Table(**table)
        self.run.log(payload)
        accepted = copy.deepcopy(self.state)
        accepted['events'][-1]['status'] = 'sdk_accepted_remote_unverified'
        accepted['status'] = 'ready'
        try:
            r.atomic_json(self.directory / 'DELIVERY.json', accepted)
        except BaseException:
            self.closed = True
            raise
        self.state = accepted

    def finish(self):
        if self.closed:
            raise RuntimeError('Already closed')
        was_ready = self.state['status'] == 'ready'
        previous = self.state['status']
        self.closed = True
        self.state['pre_finish_status'] = previous
        self.state['status'] = 'finish_uncertain'
        self.save()
        self.run.finish(exit_code=0 if was_ready else 1)
        self.state['status'] = 'client_finished_remote_unverified' if was_ready else previous
        self.save()
