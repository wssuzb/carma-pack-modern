"""Measured initialization, oracle and chain movement evidence for this toolchain."""
from pathlib import Path
import json,os,subprocess
import numpy as np
from test_car1_initialization import series,probe_input,parse_probe
from test_science import cm,v,dense
root=Path(__file__).resolve().parents[1]
records=[]
for kind in ('reported','seasonal','multiscale'):
    for seed in (0,1,2,42,82,314159,271828,4294967295):
        t,y,e=series(kind)
        raw=subprocess.check_output([os.environ['CARMA_CURRENT_PROBE'],'inspect'],input=probe_input(t,y,e,seed),text=True)
        initial=parse_probe(raw)
        theta=np.array(initial['initial'][0])
        cov=lambda dt:theta[0]**2*np.exp(-np.exp(theta[3])*dt)
        ll=dense(cov,t=t,y=y-theta[2],e=e*np.sqrt(theta[1]))[3]
        expected=ll-25/theta[1]-26*np.log(theta[1])
        cm.set_seed(seed);model=cm.run_mcmc_car1(64,64,v(t),v(y),v(e),2)
        x=np.asarray(model.getSamples());post=np.asarray(model.GetLogLikes())
        recomputed=np.array([model.getLogDensity(v(row)) for row in x])
        records.append({'series':kind,'seed':seed,'n_observations':len(t),'initial':theta.tolist(),
            'legal':bool(initial['legal'][0][0]),'initial_log_density':initial['density'][0][0],
            'cached_initial_log_density':initial['density'][0][1],
            'independent_dense_log_posterior_error':float(abs(expected-initial['density'][0][0])),
            'all_saved_posteriors_finite':bool(np.isfinite(post).all()),
            'saved_posterior_recomputation_max_abs':float(np.max(abs(post-recomputed))),
            'saved_changes_of_63':int(np.count_nonzero(np.any(np.diff(x,axis=0)!=0,axis=1))),
            'unique_saved_states':len(np.unique(x,axis=0))})
report={'purpose':'Initialization legality/cache/target consistency and non-frozen regression; NOT convergence or posterior calibration',
    'chain_settings':{'samples':64,'burnin':64,'thin':2},'records':records,
    'max_initial_posterior_oracle_error':max(r['independent_dense_log_posterior_error'] for r in records),
    'max_cached_saved_posterior_error':max(r['saved_posterior_recomputation_max_abs'] for r in records),
    'min_saved_changes_of_63':min(r['saved_changes_of_63'] for r in records)}
(root/'audit/phase2/initialization_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
