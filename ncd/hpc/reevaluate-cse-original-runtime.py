"""Read-only original-prior runtime comparison for four historical Windows cells.

No training, threshold search, original-result mutation, or equality tolerance.
Differences are retained as scientific observations, not suppressed as errors.
"""
import argparse
import collections
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def guarded_main():
    """Project stop policy: rolling60s <10% of2CPUs; not an overall90s cap."""
    assert int(os.environ.get('SLURM_CPUS_PER_TASK', '2')) == 2
    faulthandler.enable()
    stopped = threading.Event()
    samples = collections.deque([(time.monotonic(), time.process_time())])
    def watch():
        while not stopped.wait(2):
            wall, cpu = time.monotonic(), time.process_time()
            samples.append((wall, cpu))
            while len(samples) > 1 and samples[1][0] <= wall - 60:
                samples.popleft()
            elapsed = wall - samples[0][0]
            if elapsed >= 60:
                used = cpu - samples[0][1]
                low = used / (elapsed * 2) < .1
                print(json.dumps(dict(event='CPU_PROGRESS_WINDOW', window_seconds=elapsed,
                                      cpu_seconds=used, allocated_cpus=2, low=low)), flush=True)
                if low:
                    faulthandler.dump_traceback(all_threads=True)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(124)
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(90, exit=True)
    faulthandler.dump_traceback_later(90, exit=True)
    monitor = threading.Thread(target=watch, daemon=True, name='bounded-progress-watchdog')
    monitor.start()
    try:
        return main()
    finally:
        stopped.set()
        monitor.join(timeout=3)
        faulthandler.cancel_dump_traceback_later()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--forward-source', type=Path, required=True)
    p.add_argument('--sample-per-class', type=int, default=0)
    a = p.parse_args()
    assert a.sample_per_class >= 0
    assert not a.output.exists(), 'Use a fresh output directory'
    assert sha(a.forward_source) == '3fe63c8f404d5217a6c63108e7169132bab7b75751a03146d17038ecc856ca2a'
    print('RUNTIME_COMPARISON_IMPORT_BEGIN', flush=True)
    spec = importlib.util.spec_from_file_location('saved_forward', a.forward_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np, torch = module.np, module.torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print('RUNTIME_COMPARISON_IMPORT_COMPLETE', flush=True)
    arrays = []
    for key in ('train_x', 'train_y', 'test_x', 'test_y'):
        print(json.dumps(dict(event='DATA_OPEN', array=key)), flush=True)
        arrays.append(np.load(a.data / (key + '.npy'), mmap_mode='r'))
    x, y, xt, yt = arrays
    binding = {}
    if not a.sample_per_class:
        for key, array in zip(('x_train', 'y_train', 'x_test', 'y_test'), arrays):
            print(json.dumps(dict(event='DATA_HASH_BEGIN', array=key)), flush=True)
            binding[key] = dict(dtype=array.dtype.str, shape=list(array.shape),
                                sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())
            print(json.dumps(dict(event='DATA_HASH_COMPLETE', array=key)), flush=True)
    a.output.mkdir(parents=True)
    failures, completed = [], []
    for method, seed in (('icarl', 10), ('cer-bic-common', 10),
                         ('cer-bic-common', 11), ('cer-bic-common', 12)):
        try:
            root = a.inputs / 'cic-ids-2018' / method / f'seed{seed}'
            result_path = root / 'result.json'
            r = json.loads(result_path.read_text(encoding='utf-8'))
            i = r['identity']
            assert r['completed'] and i['seed'] == seed and i['class_order'][0] == 'Normal'
            assert i['dataset'] == 'CSE-CIC-IDS2018'
            assert i['method'] == ('icarl' if method == 'icarl' else 'cer-bic-flow-cnn-common')
            if not a.sample_per_class:
                assert i['data_binding'] == binding
            n = len(i['class_order'])
            counts = np.bincount(y, minlength=n)
            ids = np.flatnonzero(y == 0)
            holdout = np.random.default_rng(seed * 100003 + 17).permutation(ids)
            holdout = holdout[:max(5, int(np.ceil(len(ids) * .1)))]
            bound = i.get('section26_holdout', i.get('holdout'))['0']['validation_indices_sha256']
            assert hashlib.sha256(holdout.tobytes()).hexdigest() == bound
            if a.sample_per_class:
                holdout = holdout[:a.sample_per_class]
            assert len(holdout) * x.shape[1] * x.dtype.itemsize < 1024**3
            hx = np.empty((len(holdout), x.shape[1]), dtype=x.dtype)
            order = np.argsort(holdout)
            for offset in range(0, len(order), 8192):
                slots = order[offset:offset + 8192]
                hx[slots] = x[holdout[slots]]
            defaults = r.get('default_stages')
            if defaults is None:
                defaults = json.loads((root / 'default_stages.json').read_text(encoding='utf-8'))
            calibrated = r.get('calibrated_stages')
            if calibrated is None:
                calibrated = json.loads((root / 'calibrated_stages.json').read_text(encoding='utf-8'))
            stages = []
            assert len(r['checkpoints']) == len(defaults) == len(calibrated) == 4
            for t, cp in enumerate(r['checkpoints']):
                path = root / cp['checkpoint_file']
                assert sha(path) == cp['checkpoint_sha256']
                state = torch.load(path, map_location='cpu', weights_only=True)
                assert state['identity'] == i and state['task_index'] == t
                seen = cp['seen_classes']
                assert seen == list(range(len(seen)))
                forward = module.feature_logits(state, method, seen, int(i['effective_config']['batch_size']))
                rows = np.flatnonzero(np.isin(yt, seen))
                if a.sample_per_class:
                    rows = np.concatenate([rows[yt[rows] == c][:a.sample_per_class] for c in seen])
                def scores(array, index, batch):
                    pieces = []
                    with torch.inference_mode():
                        for off in range(0, len(index), batch):
                            values = torch.from_numpy(np.array(array[index[off:off + batch]], dtype=np.float32, copy=True))
                            z = forward(values)
                            assert torch.isfinite(z).all()
                            pieces.append(z)
                    return torch.cat(pieces)
                default_batch = 512 if method == 'cer-bic-common' else 4096
                cal_batch = 512 if method == 'cer-bic-common' else int(i['effective_config']['eval_batch_size'])
                logits = scores(xt, rows, cal_batch)
                default_logits = logits if default_batch == cal_batch else scores(xt, rows, default_batch)
                hs = scores(hx, np.arange(len(hx)), cal_batch)
                prior = torch.from_numpy(np.log(counts / counts.sum()).astype(np.float32)[seen])
                views = {}
                refs = {'default': defaults[t]['metrics']}
                refs.update(calibrated[t]['operating_points'])
                predictions = {'default': default_logits.argmax(1).numpy()}
                for alpha, point in calibrated[t]['calibration']['operating_points'].items():
                    predictions[alpha] = (logits + point['total_tau'] * prior).argmax(1).numpy()
                for view, prediction in predictions.items():
                    got = module.metrics(yt[rows], np.asarray(seen)[prediction], n, seen)
                    ref_cm = np.asarray(refs[view]['confusion_matrix'], dtype=np.int64)
                    ref_support = ref_cm.sum(1)
                    ref_bacc = float((ref_cm.diagonal()[seen] / ref_support[seen]).mean())
                    entry = dict(metrics=got, reference_confusion_matrix=ref_cm.tolist(),
                                 confusion_matrix_delta=(np.asarray(got['confusion_matrix']) - ref_cm).tolist(),
                                 balanced_accuracy_delta_pp=100 * (got['balanced_accuracy'] - ref_bacc))
                    if view != 'default':
                        point = calibrated[t]['calibration']['operating_points'][view]
                        entry['original_total_tau'] = point['total_tau']
                        entry['holdout_false_positives'] = int(((hs + point['total_tau'] * prior).argmax(1) != 0).sum())
                        entry['reference_holdout_false_positives'] = point['calibration_false_positives']
                    views[view] = entry
                stages.append(dict(task_index=t, checkpoint_sha256=cp['checkpoint_sha256'],
                                   default_batch=default_batch, calibrated_batch=cal_batch, views=views))
                print(json.dumps(dict(event='ORIGINAL_RUNTIME_STAGE', method=method, seed=seed, task=t,
                                      software_only=bool(a.sample_per_class))), flush=True)
            record = dict(method=method, seed=seed, software_only=bool(a.sample_per_class), stages=stages,
                          source_result_sha256=sha(result_path), source_code_sha256=sha(Path(__file__)),
                          forward_source_sha256=sha(a.forward_source), source_identity=i,
                          runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, platform=platform.platform()),
                          command=sys.argv, log_file='diagnosis.stdout.log',
                          scope='Read-only runtime comparison; original tau/results unchanged; sample differences are not scientific estimates.')
            (a.output / f'{method}-seed{seed}.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            completed.append(dict(method=method, seed=seed))
        except Exception as error:
            import traceback
            traceback.print_exc()
            failures.append(dict(method=method, seed=seed, error=repr(error)))
    summary = dict(completed=completed, failures=failures, software_only=bool(a.sample_per_class))
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(event='ORIGINAL_RUNTIME_COMPLETE', **summary)), flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(guarded_main())
