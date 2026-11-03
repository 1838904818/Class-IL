"""Compile the manuscript with Tectonic; reject unresolved citations/references."""
import argparse
from pathlib import Path
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tectonic', default='tectonic')
p.add_argument('--output-dir', type=Path, required=True)
a = p.parse_args()
source = Path(__file__).resolve().parent
output = a.output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
subprocess.run([a.tectonic, '-X', 'compile', 'paper.tex', '--outdir', str(output),
                '--keep-logs', '--keep-intermediates'], cwd=source, check=True)
log = (output/'paper.log').read_text(errors='replace')
errors = re.findall(r'^.*(?:undefined|multiply defined|Missing character|Overfull).*$', log, re.M)
if errors:
    raise SystemExit('\n'.join(errors))
print('Compilation and reference/overflow checks passed; visual review remains separate.')
