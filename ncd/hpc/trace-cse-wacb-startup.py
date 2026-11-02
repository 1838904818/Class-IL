"""One bounded startup diagnosis of the immutable WACB evaluator, never model inference.

Native faulthandler exits after60s in a phase, or90s overall.
The original evaluator bytes, hashing result and input data are unchanged.
"""
import argparse,ast,faulthandler,hashlib,json,os,runpy,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
target=a.target.resolve(strict=True);raw=target.read_bytes();assert hashlib.sha256(raw).hexdigest()=='736c77dc6a926a4c1591b1caf25b646c235e384237952dcbed960ae3e522296c'
tree=ast.parse(raw);started=time.monotonic();seen=set();real_sha=hashlib.sha256;timeout=60;global_limit=90
def event(name,**fields):
    print(json.dumps(dict(event=name,elapsed_seconds=time.monotonic()-started,**fields)),flush=True)
def arm(name):
    remaining=global_limit-(time.monotonic()-started)
    if remaining<=0:faulthandler.dump_traceback();os._exit(124)
    limit=min(timeout,remaining)
    faulthandler.cancel_dump_traceback_later();faulthandler.dump_traceback_later(limit,repeat=False,exit=True);event('BOUNDED_PHASE',phase=name,stall_exit_seconds=limit,global_limit_seconds=global_limit)
faulthandler.enable();markers={}
for node in tree.body:
    if isinstance(node,(ast.Import,ast.ImportFrom)):markers[node.lineno]='IMPORT '+ast.get_source_segment(raw.decode(),node)
main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
stop_line=None
for node in ast.walk(main):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
        if node.func.value.id=='torch' and node.func.attr=='load':stop_line=node.lineno
    if isinstance(node,ast.Assign):
        names=[v.id for v in node.targets if isinstance(v,ast.Name)]
        for name in names:
            if name in ('data','binding','counts','truth','root'):markers[node.lineno]='PRE_MODEL '+name
assert stop_line is not None
def sha(data=b'',*args,**kwargs):
    large=len(data)>=1024*1024
    if large:arm('SHA256 '+str(len(data))+' payload bytes')
    result=real_sha(data,*args,**kwargs)
    if large:event('LARGE_SHA256_FINISHED',payload_bytes=len(data));arm('AFTER_LARGE_SHA256')
    return result
def trace(frame,kind,arg):
    if frame.f_code.co_filename!=str(target):return None
    if kind=='line':
        line=frame.f_lineno
        if line==stop_line:
            event('STARTUP_PREPARED_BINDING_CHECK_COMPLETED_NO_MODEL_LOAD_OR_INFERENCE',source_line=line)
            faulthandler.cancel_dump_traceback_later();sys.stdout.flush();sys.stderr.flush();os._exit(0)
        if line in markers and line not in seen:
            seen.add(line);arm(markers[line]);event('SOURCE_LINE',line=line)
    return trace
event('STARTUP_TRACE_BEGIN',target_sha256=real_sha(raw).hexdigest(),wrapper_sha256=real_sha(Path(__file__).read_bytes()).hexdigest(),scope='Startup/import/read/hash only; stops before the first checkpoint load, no model inference or training.')
hashlib.sha256=sha;sys.settrace(trace);arm('RUN_IMMUTABLE_EVALUATOR')
arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
sys.argv=[str(target),*arguments];runpy.run_path(str(target),run_name='__main__')
raise RuntimeError('Unexpected return before the startup-only stop boundary')
