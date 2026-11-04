"""Emit actual maximum errors, using the installed wheel and independent oracles."""
import json, os, subprocess
import numpy as np
from test_science import cm,v,T,Y,E,dense,state_cov,parse,run
metrics={}
k=cm.KalmanFilter1(v(T),v(Y),v(E),.7,.3);k.Filter()
m,var,p,ll=dense(lambda dt:.7/.6*np.exp(-.3*dt))
metrics['OU_mean_max_abs']=float(np.max(np.abs(np.array(k.GetMean())-m)))
metrics['OU_var_max_abs']=float(np.max(np.abs(np.array(k.GetVar())-var)))
metrics['OU_prediction_max_abs']=float(np.max(np.abs(np.array([[k.Predict(x).first,k.Predict(x).second] for x in (-1,0,1,4,12)])-p)))
for name,roots,ma in [('CAR2',[-1.,-.3],[1.,0.]),('CARMA21',[-.4+1j,-.4-1j],[1.,.7]),('CARMA32',[-1.,-.3,-2.],[1.,.4,.1])]:
    k=cm.KalmanFilterp(v(T),v(Y),v(E),.7,cm.vecC(roots),v(ma));k.Filter()
    m,var,p,ll=dense(state_cov(roots,ma,.7))
    metrics[name+'_mean_max_abs']=float(np.max(np.abs(np.array(k.GetMean())-m)))
    metrics[name+'_var_max_abs']=float(np.max(np.abs(np.array(k.GetVar())-var)))
    metrics[name+'_prediction_max_abs']=float(np.max(np.abs(np.array([[k.Predict(x).first,k.Predict(x).second] for x in (-1,0,1,4,12)])-p)))
ref=parse(subprocess.check_output([os.environ['CARMA_REFERENCE_DRIVER']],text=True))
for q in (-1,0,1):
    model=run(q)
    metrics['chain'+str(q+2)+'_max_abs']=float(np.max(np.abs(np.array(model.getSamples())-ref['chain'+str(q+2)])))
    metrics['log'+str(q+2)+'_max_abs']=float(np.max(np.abs(np.array(model.GetLogLikes())-np.array(ref['log'+str(q+2)])[:,0])))
from pathlib import Path
Path(os.environ.get('CARMA_METRICS_DIRECTORY',str(Path(__file__).resolve().parents[1]/'audit'))).joinpath('numerical_metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
