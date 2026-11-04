"""Historical defect remains reproducible in the immutable upstream oracle."""
import os,subprocess
import numpy as np
from test_car1_initialization import series,probe_input,parse_probe

def test_reproduce_upstream_car1_invalid_random_start():
    t,y,e=series('reported')
    raw=subprocess.check_output([os.environ['CARMA_UPSTREAM_PROBE'],'auto'],input=probe_input(t,y,e,314159,12,20,2),text=True,timeout=10)
    result=parse_probe(raw)
    assert np.isneginf(result['posterior']).all()
    x=np.array(result['sample'])
    assert np.all(x==x[0])
