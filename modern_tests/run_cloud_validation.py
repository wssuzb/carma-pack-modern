"""Capture pytest output via a pipe, then persist it with exit status (no mixed stdio file offsets)."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
audit=root/'audit/phase2'
args=[sys.executable,'-m','pytest',str(root/'modern_tests'),'-v','--capture=tee-sys','--junitxml='+str(audit/'pytest.xml')]
r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
(audit/'pytest.log').write_text(r.stdout)
(audit/'validation_command.json').write_text(json.dumps({'command':args,'exit_code':r.returncode,'python':sys.executable,'drivers':{k:os.environ[k] for k in ['CARMA_REFERENCE_DRIVER','CARMA_UPSTREAM_PROBE','CARMA_CURRENT_PROBE']}},indent=2))
print(r.stdout.splitlines()[-1]);sys.exit(r.returncode)
