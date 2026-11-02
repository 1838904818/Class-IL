"""Observe the immutable WACB evaluation and stop sustained low process-CPU use.

No startup-only stop or fixed90s overall cap. A complete rolling60s window
below10% of two requested CPUs exits124 with a stack. A rearmed native90s
watchdog protects the monitor itself; normal CPU-progress work can run longer.
"""
import argparse,ast,collections,faulthandler,hashlib,json,os,runpy,sys,threading,time
from pathlib import Path

class CpuWindow:
    def __init__(self,cpus=2,seconds=60,minimum=.10):self.cpus=cpus;self.seconds=seconds;self.minimum=minimum;self.samples=collections.deque()
    def sample(self,wall,cpu):
        self.samples.append((wall,cpu));cutoff=wall-self.seconds
        while len(self.samples)>1 and self.samples[1][0]<=cutoff:self.samples.popleft()
        first,used=self.samples[0];elapsed=wall-first
        if elapsed<self.seconds:return None
        utilization=max(0.,cpu-used)/(elapsed*self.cpus)
        return dict(window_seconds=elapsed,cpu_seconds=max(0.,cpu-used),allocated_cpus=self.cpus,utilization_fraction=utilization,low=utilization<self.minimum)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',type=Path);p.add_argument('--self-test',action='store_true');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.self_test:
        g=CpuWindow(seconds=4);assert g.sample(0,0) is None;assert g.sample(3,.01) is None;assert g.sample(6,.02)['low']
        g=CpuWindow(seconds=4)
        for t in range(201):
            value=g.sample(float(t),float(t))
            if value is not None:assert not value['low']
        g=CpuWindow();g.sample(0,0);assert g.sample(60,2.484*60/1009)['low']
        print('CPU_WINDOW_IDLE_BUSY_IRREGULAR_AND_LONG_PROGRESS_SOFTWARE_OK');return
    target=a.target.resolve(strict=True);raw=target.read_bytes();real_sha=hashlib.sha256;assert real_sha(raw).hexdigest()=='736c77dc6a926a4c1591b1caf25b646c235e384237952dcbed960ae3e522296c'
    assert int(os.environ.get('SLURM_CPUS_PER_TASK','2'))==2
    tree=ast.parse(raw);started=time.monotonic();stopped=threading.Event();seen=set();phase='wrapper';window=CpuWindow();window.sample(started,time.process_time());markers={}
    def event(name,**fields):print(json.dumps(dict(event=name,elapsed_seconds=time.monotonic()-started,**fields)),flush=True)
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):markers[node.lineno]='IMPORT '+ast.get_source_segment(raw.decode(),node)
    body=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='main')
    for node in ast.walk(body):
        if isinstance(node,ast.Assign):
            for name in [v.id for v in node.targets if isinstance(v,ast.Name)]:
                if name in ('data','binding','counts','truth','root','path','h','t','hs','ts','benign','record','stem'):markers[node.lineno]=name
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='np' and node.func.attr=='savez_compressed':markers[node.lineno]='SAVE_COMPRESSED_ARRAYS'
    faulthandler.enable()
    def watchdog():
        while not stopped.wait(2):
            value=window.sample(time.monotonic(),time.process_time())
            if value is not None:
                event('CPU_PROGRESS_WINDOW',phase=phase,**value)
                if value['low']:
                    event('LOW_CPU_STALL_STOP',phase=phase,exit_code=124);faulthandler.dump_traceback(all_threads=True);sys.stdout.flush();sys.stderr.flush();os._exit(124)
            faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(90,repeat=False,exit=True)
    def sha(data=b'',*args,**kwargs):
        large=len(data)>=1024*1024
        if large:event('LARGE_SHA256_BEGIN',payload_bytes=len(data))
        result=real_sha(data,*args,**kwargs)
        if large:event('LARGE_SHA256_FINISHED',payload_bytes=len(data))
        return result
    def trace(frame,kind,arg):
        nonlocal phase
        if frame.f_code.co_filename!=str(target) or frame.f_code.co_name not in ('<module>','main'):return None
        if kind=='line' and frame.f_lineno in markers:
            seed=frame.f_locals.get('seed');key=(frame.f_lineno,seed)
            if key not in seen:
                seen.add(key);phase=markers[frame.f_lineno];event('SOURCE_PHASE',phase=phase,source_line=frame.f_lineno,seed=seed)
        return trace
    event('GUARDED_EVALUATION_BEGIN',target_sha256=real_sha(raw).hexdigest(),guard_sha256=real_sha(Path(__file__).read_bytes()).hexdigest(),cpu_window_seconds=60,minimum_allocated_cpu_fraction=.10,allocated_cpus=2,outer_command=sys.argv,scope='Observability and resource-stop only; original evaluation/model/float32 decisions unchanged.')
    faulthandler.dump_traceback_later(90,repeat=False,exit=True);monitor=threading.Thread(target=watchdog,name='bounded-cpu-progress-watchdog',daemon=True);monitor.start();hashlib.sha256=sha;sys.settrace(trace)
    arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments;sys.argv=[str(target),*arguments]
    try:runpy.run_path(str(target),run_name='__main__')
    finally:
        stopped.set();monitor.join(timeout=3);faulthandler.cancel_dump_traceback_later();sys.settrace(None);event('GUARDED_EVALUATION_EXIT',phase=phase)
if __name__=='__main__':main()
