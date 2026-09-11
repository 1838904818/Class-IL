"""Bounded own-step accounting; no scheduler loop, discovery or foreign jobs."""
import math
import re
import subprocess
from decimal import Decimal
from pathlib import Path
from chunk_core import require


def memory_bytes(value):
    m=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',value)
    require(m is not None,'missing/invalid memory metric')
    scale=1024**('KMGT'.index(m[2])+1) if m[2] else 1
    n=Decimal(m[1])*scale
    require(n==int(n),'fractional byte metric')
    return int(n)


def cpu_seconds(value):
    m=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2}(?:\.\d+)?)',value)
    require(m is not None,'missing/invalid CPU time')
    days,hours,minutes,seconds=m.groups()
    require(int(minutes)<60 and Decimal(seconds)<60 and (days is None or int(hours)<24),
            'CPU clock component out of range')
    return float(int(days or 0)*86400+int(hours)*3600+int(minutes)*60+Decimal(seconds))


def step_identity(job,step):
    require(re.fullmatch(r'[1-9]\d*',job) is not None and re.fullmatch(r'\d+',step) is not None,
            'numeric own computation step required')
    return job+'.'+step


def parse_sstat(text,expected):
    lines=text.strip().splitlines()
    require(len(lines)==1,'one accounting row required')
    fields=[f.strip() for f in lines[0].split('|')]
    require(len(fields)==5 and fields[0]==expected and fields[4]=='1','step/task identity mismatch')
    maximum,average=map(memory_bytes,fields[1:3])
    require(maximum>=average,'RSS ordering mismatch')
    return dict(source='slurm-step-observation',step=expected,max_rss_bytes=maximum,
        ave_rss_bytes=average,cpu_seconds=cpu_seconds(fields[3]),raw_fields=fields)


class LimitedSstat:
    """Three calls total, including at most one initial retry; failures are spent."""
    def __init__(self,job,step,run=subprocess.run):
        self.identity=step_identity(job,step);self.run=run;self.calls=0
        self.events=set()
    def sample(self,event):
        require(event in ('initial','initial_retry','midpoint','terminal'),'accounting event')
        require(event not in self.events and self.calls<3,'accounting RPC budget spent')
        require(event!='initial_retry' or 'initial' in self.events,'retry before initial')
        self.events.add(event);self.calls+=1
        proc=self.run(['sstat','--noheader','--parsable2','--jobs='+self.identity,
            '--format=JobID,MaxRSS,AveRSS,AveCPU,NTasks'],capture_output=True,text=True,
            timeout=12,check=True)
        return parse_sstat(proc.stdout,self.identity)


def interval_cpu(previous,current,seconds,cpus=2):
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (previous,current,seconds)),
            'nonfinite CPU sample')
    require(previous>=0 and current>=previous and seconds>0 and cpus==2,'CPU counter/time reset')
    return 100*(current-previous)/(seconds*cpus)


def own_cgroup_v2(proc_text,job,step,root=Path('/sys/fs/cgroup')):
    """Resolve only the supplied /proc/self/cgroup v2 entry; never enumerate."""
    step_identity(job,step)
    rows=[line[3:] for line in proc_text.splitlines() if line.startswith('0::')]
    require(len(rows)==1,'own cgroup v2 entry unavailable')
    relative=Path(rows[0].lstrip('/'))
    require('..' not in relative.parts and '\\' not in rows[0],'unsafe cgroup path')
    require('job_'+job in relative.parts and 'step_'+step in relative.parts,'cgroup is not exact own step')
    path=root/relative
    require(path.resolve().is_relative_to(root.resolve()),'cgroup escape')
    require(all(not p.is_symlink() for p in (path,*path.parents)),'linked cgroup path')
    return path


def cgroup_sample(path):
    """Only call on the exact path returned by own_cgroup_v2."""
    values={}
    for line in (path/'cpu.stat').read_text().splitlines():
        key,value=line.split()
        require(key not in values,'duplicate cgroup metric');values[key]=value
    require(re.fullmatch(r'\d+',values.get('usage_usec','')) is not None,'cgroup CPU unavailable')
    current=(path/'memory.current').read_text().strip()
    peak=(path/'memory.peak').read_text().strip()
    require(current.isdecimal() and peak.isdecimal() and int(peak)>=int(current),'cgroup memory unavailable')
    return dict(source='own-step-cgroup-v2',cpu_seconds=int(values['usage_usec'])/1e6,
        memory_current_bytes=int(current),memory_peak_bytes=int(peak))


class Window:
    """Fixed 60-second rolling window; 300s grace then 240s sustained low use.

    A reason blocks the NEXT record. Hard per-record time/memory limits remain
    the supervisor's independent obligation. Missing telemetry is never zero.
    """
    def __init__(self):self.samples=[];self.low_since=None;self.last_time=None
    def update(self,elapsed,cpu,gpu):
        require(all(type(x) in (int,float) and math.isfinite(x) for x in (elapsed,cpu,gpu)),
                'invalid utilization')
        require(elapsed>=0 and cpu>=0 and 0<=gpu<=100,'utilization range')
        require(self.last_time is None or elapsed>self.last_time,'nonmonotonic monitor time')
        gap=self.last_time is not None and elapsed-self.last_time>30
        self.last_time=elapsed
        self.samples.append((elapsed,cpu,gpu))
        while len(self.samples)>1 and self.samples[1][0]<=elapsed-60:self.samples.pop(0)
        if gap:
            return 'MONITORING_INSUFFICIENT'
        if elapsed<300 or elapsed-self.samples[0][0]<60:return None
        durations=[self.samples[i][0]-max(self.samples[i-1][0],elapsed-60)
                   for i in range(1,len(self.samples))]
        total=sum(durations)
        avgs=[sum(d*self.samples[i][k] for i,d in enumerate(durations,1))/total for k in (1,2)]
        low=min(avgs)<10
        self.low_since=(self.low_since if self.low_since is not None else elapsed) if low else None
        return 'LOW_UTILIZATION' if self.low_since is not None and elapsed-self.low_since>=240 else None
