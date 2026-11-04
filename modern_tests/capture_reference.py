"""Capture only recognized scientific output; preserve complete raw output separately."""
import json,subprocess,sys
from pathlib import Path
result=subprocess.run([sys.argv[1]],text=True,capture_output=True,check=True)
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
(out/'reference_raw.txt').write_text(result.stdout+result.stderr)
known={'mean','var','pred','chain1','chain2','chain3','log1','log2','log3'}
values={}
for line in result.stdout.splitlines():
    parts=line.split()
    if parts and parts[0] in known:values.setdefault(parts[0],[]).append([float(x) for x in parts[1:]])
for name in ('chain1','chain2','chain3'):
    assert len(values.get(name,[]))==12,(name,values.keys())
(out/'reference.json').write_text(json.dumps(values,indent=2,allow_nan=False))
print('Captured', {k:len(v) for k,v in values.items()})
