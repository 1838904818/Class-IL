"""Explicit online destination policy; pure validation, no SDK or credentials."""
import re


def init_arguments(policy, *, run_id, resume):
    if not isinstance(policy, dict) or set(policy) != {'entity', 'project', 'aggregate_only', 'allow_online'}:
        raise ValueError('Explicit tracking policy required')
    if policy['aggregate_only'] is not True or policy['allow_online'] is not True:
        raise ValueError('Aggregate online approval required')
    for key in ('entity', 'project'):
        if not isinstance(policy[key], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', policy[key]):
            raise ValueError('Invalid explicit destination')
    if not isinstance(run_id, str) or not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise ValueError('Persisted opaque run identity required')
    if type(resume) is not bool:
        raise ValueError('Explicit recovery mode required')
    return dict(entity=policy['entity'], project=policy['project'], id=run_id,
                mode='online', force=True, resume='must' if resume else 'never',
                reinit='create_new', save_code=False,
                config={'stage': 'L1_head_loss_control', 'aggregate_only': True})
