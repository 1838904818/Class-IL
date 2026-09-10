"""Bounded context-adapter diagnostics, never an ETG efficacy or SHAP estimate."""
import time
import numpy as np
from native_target import CallBudget, FixedContextTarget, array_sha, capture_native, validate_native
from pilot_core import require


class Mismatch(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__('bounded profile native mismatch')


def exact_scores(expected, actual, label):
    for key in ('class_axis', 'head_scores', 'router_z_scores', 'joint_scores', 'predicted_class_id'):
        a, b = np.asarray(expected[key]), np.asarray(actual[key])
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            first = None
            if a.shape == b.shape:
                ix = np.argwhere(a != b)
                first = ix[0].tolist() if len(ix) else None
            raise Mismatch({'check': label, 'component': key, 'first_index': first,
                'expected_sha256': array_sha(a), 'actual_sha256': array_sha(b),
                'expected_shape': list(a.shape), 'actual_shape': list(b.shape)})


def masks(row, references):
    require(row.shape == (78,) and references.shape == (32, 78), 'profile perturbation contract')
    cases = [('unmasked', row.copy()), ('all-reference-first', references[0].copy()),
             ('all-reference-last', references[-1].copy())]
    for name, indices, ref in [('first-feature', [0], 0), ('last-feature', [77], 31),
                               ('alternating-features', np.arange(0, 78, 2), 16)]:
        x = row.copy(); x[indices] = references[ref, indices]; cases.append((name, x))
    return cases


def profile(models, inputs, limits, synchronize=lambda: None, on_record=lambda r: None):
    """models is a sequential generator to avoid retaining two GPU models."""
    budget = CallBudget(limits['max_native_calls'], limits['native_seconds'], synchronize)
    records = []; checkpoints = []
    for checkpoint, model in models:
        require(checkpoint == len(checkpoints) and checkpoint in (0, 1), 'checkpoint order')
        axis = [0, 1] if checkpoint == 0 else [0, 1, 2, 3]
        require(model.metadata['seen_classes'] == axis, 'seen-class axis')
        checkpoints.append(checkpoint)
        for item in inputs.targets:
            cid = item['class_id']; parent = inputs.contexts[cid]; row = item['row_in_parent']
            parent_hash = array_sha(parent)
            first = validate_native(budget.run(model.score, parent.copy()), len(parent), axis)
            again = validate_native(budget.run(model.score, parent.copy()), len(parent), axis)
            exact_scores(first, again, 'unmasked-repeat')
            captured = budget.run(lambda x: capture_native(model, x), parent.copy())
            exact_scores(first, captured, 'same-forward-logit-hook')
            target = FixedContextTarget(model.score, parent, row, axis, cid, budget)
            for name, replacement in masks(parent[row], inputs.references):
                raw = parent.copy(); raw[row] = replacement
                before = array_sha(raw)
                begin = time.monotonic()
                direct = validate_native(budget.run(model.score, raw), len(raw), axis)
                require(array_sha(raw) == before, 'native scorer changed profile input')
                repeat = validate_native(budget.run(model.score, raw.copy()), len(raw), axis)
                exact_scores(direct, repeat, 'masked-repeat:' + name)
                j = axis.index(cid)
                joint = direct['joint_scores'][row]
                expected = float(joint[j] - np.max(np.delete(joint, j)))
                actual = target(replacement[None, :])[0]
                if actual != expected:
                    raise Mismatch({'check': 'wrapper-native-margin', 'checkpoint': checkpoint,
                                    'class_id': cid, 'mask': name})
                record = {'checkpoint': checkpoint, 'class_id': cid, 'mask': name,
                    'context_sha256': parent_hash, 'perturbed_context_sha256': before,
                    'row_id': item['row_id'], 'parent_rows': len(parent),
                    'native_scores_sha256': array_sha(direct['joint_scores']),
                    'exact_repeat_and_adapter_match': True, 'elapsed_seconds': time.monotonic()-begin}
                records.append(record); on_record(record)
            require(array_sha(parent) == parent_hash, 'frozen parent modified')
            del target, first, again, captured, direct, repeat
        # The generator must release this model before allocating the next one.
        del model
    require(checkpoints == [0, 1] and len(records) == 24, 'incomplete diagnostic workload')
    return {'schema': 'etg-native-component-profile-v1', 'status': 'COMPONENT_PROFILE_COMPLETE',
        'records': records, 'native_calls': budget.calls, 'native_context_rows': budget.context_rows,
        'elapsed_profile_window_seconds': time.monotonic()-budget.start,
        'archived_score_parity_verified': False, 'full_population_parity_verified': False,
        'shap_attributions_computed': False, 'etg_actions_fitted': False,
        'efficacy_metrics_computed': False, 'scientific_result_generated': False,
        'interpretation': 'Same-current-environment adapter/repeat check; not historical numerical parity or a SHAP/ETG study.'}
