"""CIC/CSE entry for the unchanged matched-protocol ECBRS paper-memory adaptation.

This does not reproduce the authors' binary training entry. The registered
multiclass MLP, task schedule, permanent benign holdout and evaluation apply.
The literal memory algorithm and all adaptation choices are saved in identity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['NCD_FROZEN_ROOT']).resolve(strict=True)
sys.path.insert(0, str(ROOT))
if os.name != 'nt' and os.environ.get('NCD_SKLEARN_ROOT'):
    sys.path.insert(1, os.environ['NCD_SKLEARN_ROOT'])
import numpy as np
import torch
from ncd import run_problem_scout as ps, run_replay_bs as bs
from ncd.crr_benchmarks import training_epoch_batches, _class_balanced_cross_entropy
from ncd.default_decision import evaluate_models
from ncd.ecbrs_memory import PaperECBRS, SEMANTICS
from ncd.model import MLP
from ncd.prepare_problem_scout_data import prepare_setting
from ncd.source_commit import source_commit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['NSL-KDD', 'CIC-IDS-2017', 'CSE-CIC-IDS2018'], required=True)
    p.add_argument('--seed', type=int, choices=range(10, 15), required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--log-prefix', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    wd = Path(os.environ['NCD_CACHE_ROOT']).resolve(strict=True)
    assert a.data_root.resolve(strict=True) == Path(os.environ['NCD_DATA_ROOT']).resolve(strict=True)
    assert a.output_dir.resolve().is_relative_to(Path(os.environ['NCD_OUTPUT_ROOT']).resolve())
    os.environ['TEMP'] = os.environ['TMP'] = str(wd/'tmp')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    selection = json.loads((ROOT/'paper_aacil/results/section123_schedule_selection.json').read_text(encoding='utf-8'))
    if selection['selected_schedule'] != 'original_fixed':
        p.error('Registered original_fixed schedule required')
    config = json.loads((ROOT/'ncd/configs/problem_scout_section21.json').read_text(encoding='utf-8'))
    config.update(problem_scout_data_root=str(a.data_root.resolve()), hidden=128, hidden_layers=2,
        device='cpu', cpu_threads=2, epochs_per_task=1 if a.smoke else 8,
        min_steps_per_task=0 if a.smoke else 2000, learning_rate=.001, batch_size=256,
        smoke=a.smoke, buffer_per_class=200)
    data, _, _, metadata = prepare_setting(a.dataset, config)
    original_counts = np.bincount(data.y_train, minlength=len(data.class_order))
    if a.smoke:
        data, _, _ = ps.smoke_subset(data, None, None, limit=64)
    train, valid = ps.section26_pools(data, a.seed)
    source_files = ['ncd/run_section178_ecbrs_extended.py', 'ncd/run_section178_ecbrs.py', 'ncd/ecbrs_memory.py',
        'ncd/run_problem_scout.py', 'ncd/run_replay_bs.py', 'ncd/crr_benchmarks.py',
        'ncd/default_decision.py', 'ncd/model.py', 'ncd/prepare_problem_scout_data.py']
    identity = dict(git_commit=source_commit(ROOT), dataset=a.dataset, method='ecbrs-paper-literal',
        seed=a.seed, smoke=a.smoke, portable_entry_sha256=ps.digest(Path(__file__)), entry_adaptation='NSL portable dataset extension; unchanged registered multiclass training and paper-memory loop.', effective_config=config, tasks=data.tasks,
        class_order=data.class_order, data_binding=ps.binding(data, None, None),
        data_metadata_sha256=ps.canonical_hash(metadata),
        command=[v for v in sys.argv if v != '--resume'],
        source_sha256={f: ps.digest(ROOT/f) for f in source_files},
        original_train_counts=original_counts.tolist(), memory_semantics=SEMANTICS,
        implementation_scope='Paper Algorithm 1 and literal Equation 1 memory, adapted to the registered multiclass protocol; not author-code execution or exact native binary reproduction.',
        author_code='https://github.com/amalapuram/CLbasedNIDS/tree/0fa7ac97ff441a865ed89aad69f05fc7e2eb7840',
        author_difference='Author memory_update uses minority_allocation and different full/victim/retrieval rules; the completed UNSW author-rule sensitivity is separate and is not executed here.',
        adaptations=['Same 2x128 MLP, Adam, eight epochs / minimum 2000 steps per task as the registered neural comparison.',
            'A single class-balanced multiclass CE on concatenated stream and inverse-local-count replay; native binary BCE is not used.',
            'Each current-task presentation is a stream encounter, including epoch repeats; replay is not re-counted or re-admitted.',
            'Budget grows to 200 times seen classes including Normal; old discarded examples are never replenished.',
            'Raw-count softmax gamma follows the printed equation; illustrative gamma=262 is not claimed reproduced.'],
        holdout={str(c): dict(train_count=len(train[c]), validation_count=len(valid[c]),
            validation_indices_sha256=hashlib.sha256(valid[c].tobytes()).hexdigest()) for c in valid},
        log_paths=[str(a.log_prefix.resolve())+f'.{s}.log' for s in ('stdout','stderr')])
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=a.resume)
    if (out/'identity.json').exists():
        assert json.loads((out/'identity.json').read_text()) == identity, 'Resume identity mismatch'
    else:
        ps.save_json(out/'identity.json', identity)
    if (out/'result.json').exists():
        completed = json.loads((out/'result.json').read_text(encoding='utf-8'))
        assert completed['completed'] and completed['identity'] == identity
        assert len(completed['checkpoints']) == len(data.tasks)
        for cp in completed['checkpoints']:
            assert ps.digest(out/cp['checkpoint_file']) == cp['checkpoint_sha256']
        assert json.loads((out/'summary.json').read_text())['completed']
        print('CELL_RESUME_COMPLETE_VALIDATED_NO_OVERWRITE', flush=True)
        return
    started_all = time.perf_counter()
    memory = PaperECBRS(200*len(data.tasks[0]), seed=a.seed)
    ps.seed_everything(a.seed*10000)
    model = MLP(data.x_train.shape[1], 128, len(data.class_order))
    records, defaults, calibrated = [], [], []
    cumulative = 0.
    for t, task in enumerate(data.tasks):
        seen = [c for part in data.tasks[:t+1] for c in part]
        task_seed = a.seed*10000+t
        ps.seed_everything(task_seed)
        checkpoint = out/f'checkpoints/checkpoint_{t:03d}.pt'
        if checkpoint.exists():
            state = ps.checkpoint_load(checkpoint, identity)
            assert state['task_index'] == t and state['seen_classes'] == seen
            model.load_state_dict(state['model_state'])
            memory = PaperECBRS.restore(state['memory'])
            cumulative = state['training_seconds_cumulative']
            print(f'CHECKPOINT_RESUME task={t}', flush=True)
        else:
            memory.grow(200*len(seen))
            current = np.sort(np.concatenate([train[c] for c in task]))
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            stats = {}
            before = len(memory.rows)
            start = time.perf_counter()
            print(f'CHECKPOINT_TRAINING task={t} current_rows={len(current)} replay_rows={before}', flush=True)
            for epoch, batches in training_epoch_batches(current[:, None], data.y_train[current], 256,
                    task_seed, config['epochs_per_task'], config['min_steps_per_task'], stats):
                model.train()
                for ids, labels in batches:
                    current_ids = ids[:, 0].numpy()
                    replay_ids, replay_labels = memory.sample(256)
                    assert np.array_equal(data.y_train[replay_ids], replay_labels)
                    joined = np.concatenate([current_ids, replay_ids])
                    x = torch.from_numpy(np.asarray(data.x_train[joined]))
                    y = torch.from_numpy(np.asarray(data.y_train[joined]))
                    optimizer.zero_grad(set_to_none=True)
                    loss = _class_balanced_cross_entropy(model(x), y)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    loss.backward()
                    optimizer.step()
                    for row, label in zip(current_ids, labels.numpy()):
                        memory.offer(int(row), int(label))
                print(f'EPOCH_DONE task={t} epoch={epoch+1} memory_rows={len(memory.rows)}', flush=True)
            seconds = time.perf_counter()-start
            cumulative += seconds
            state = dict(identity=identity, task_index=t, seen_classes=seen,
                model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                memory=memory.state(), training_seconds=seconds, training_seconds_cumulative=cumulative,
                training_schedule=stats, replay_examples_before=before, replay_examples_after=len(memory.rows),
                current_rows_sha256=hashlib.sha256(current.tobytes()).hexdigest(), current_rows=len(current),
                memory_class_counts=memory.counts(), memory_gamma=memory.gamma())
            ps.checkpoint_save(checkpoint, state)
        # Explicitly validate permanent holdout exclusion and memory ownership.
        rows = np.asarray(memory.rows, dtype=np.int64)
        held = np.concatenate(list(valid.values()))
        assert not np.intersect1d(rows, held).size and len(rows) <= 200*len(seen)
        assert np.array_equal(data.y_train[rows], np.asarray(memory.labels))
        assert set(memory.labels).issubset(seen)
        model.eval()
        default = evaluate_models(data.x_test, data.y_test, seen, data.class_order, model)
        calibration, points, cal_seconds, eval_seconds = bs.evaluate(model, data, seen,
            valid[data.class_order.index('Normal')], original_counts, config, torch.device('cpu'))
        cp_hash = ps.digest(checkpoint)
        record = {k: state[k] for k in ('task_index', 'seen_classes', 'training_seconds',
            'training_seconds_cumulative', 'training_schedule', 'replay_examples_before', 'replay_examples_after',
            'current_rows_sha256', 'current_rows', 'memory_class_counts', 'memory_gamma')}
        record.update(checkpoint_file=checkpoint.relative_to(out).as_posix(), checkpoint_sha256=cp_hash,
            identity_hash=ps.canonical_hash(identity))
        ps.save_json(checkpoint.with_suffix('.json'), record)
        records.append(record)
        defaults.append(dict(task_index=t, checkpoint_sha256=cp_hash, metrics=default))
        calibrated.append(dict(task_index=t, checkpoint_sha256=cp_hash, calibration=calibration,
            operating_points=points, calibration_seconds=cal_seconds, evaluation_seconds=eval_seconds))
        print(f'CHECKPOINT_EVALUATED task={t}', flush=True)
    ps.save_json(out/'result.json', dict(completed=True, scientific_result=not a.smoke,
        identity=identity, checkpoints=records, default_stages=defaults, calibrated_stages=calibrated,
        training_seconds_total=cumulative))
    ps.save_json(out/'summary.json', dict(completed=True, scientific_result=not a.smoke,
        method=identity['method'], seed=a.seed, smoke=a.smoke, checkpoints=len(records),
        elapsed_seconds=time.perf_counter()-started_all, training_seconds_total=cumulative))
    print('CELL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
