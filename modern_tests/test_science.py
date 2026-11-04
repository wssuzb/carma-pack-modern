"""Independent dense-Gaussian and state-space oracles, plus raw C++ regression."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.signal import tf2ss
import carmcmc as cm
v=cm.vecD
T=np.array([0,.4,1.7,2.1,4.8,7,10])
Y=np.array([.2,-.1,.5,.3,-.4,.1,.2])
E=np.array([.1,.15,.1,.2,.12,.1,.15])

def dense(cov,t=T,y=Y,e=E,queries=(-1,0,1,4,12)):
    K=np.array([[cov(abs(a-b)) for b in t] for a in t])+np.diag(e**2)
    means=[];variances=[]
    for i in range(len(t)):
        means.append(K[i,:i]@np.linalg.solve(K[:i,:i],y[:i]) if i else 0)
        variances.append(K[i,i]-(K[i,:i]@np.linalg.solve(K[:i,:i],K[:i,i]) if i else 0))
    pred=[]
    for x in queries:
        k=np.array([cov(abs(x-a)) for a in t])
        pred.append([k@np.linalg.solve(K,y),cov(0)-k@np.linalg.solve(K,k)])
    ll=-.5*(np.linalg.slogdet(K)[1]+y@np.linalg.solve(K,y)) # upstream omits n log(2pi)
    return means,variances,pred,ll

def state_cov(roots,ma,sigsqr):
    # Independent companion realization of beta(s)/alpha(s), not upstream residues/rotated Kalman.
    A,B,C,D=tf2ss(np.trim_zeros(list(ma)[::-1],'f'),np.poly(roots).real)
    P=solve_continuous_lyapunov(A,-sigsqr*B@B.T)
    return lambda lag: float((C@expm(A*lag)@P@C.T).item())

@pytest.mark.parametrize('s,o',[(.7,.3),(.03,1.2),(4,.07)])
def test_ou_dense(s,o):
    k=cm.KalmanFilter1(v(T),v(Y),v(E),s,o);k.Filter()
    m,var,p,ll=dense(lambda dt:s/(2*o)*np.exp(-o*dt))
    assert_allclose(k.GetMean(),m,rtol=2e-12,atol=2e-12)
    assert_allclose(k.GetVar(),var,rtol=2e-12,atol=2e-12)
    assert_allclose([[k.Predict(x).first,k.Predict(x).second] for x in (-1,0,1,4,12)],p,rtol=2e-11,atol=2e-12)

@pytest.mark.parametrize('roots,ma',[
    ([-1.,-.3],[1.,0.]),([-1.,-.3],[1.,.4]),
    ([-.4+1j,-.4-1j],[1.,.7]),([-1.,-.3,-2.],[1.,.4,.1])])
def test_carmap_state_space(roots,ma):
    s=.7;k=cm.KalmanFilterp(v(T),v(Y),v(E),s,cm.vecC(roots),v(ma));k.Filter()
    m,var,p,ll=dense(state_cov(roots,ma,s))
    assert_allclose(k.GetMean(),m,rtol=2e-10,atol=2e-11)
    assert_allclose(k.GetVar(),var,rtol=2e-10,atol=2e-11)
    assert_allclose([[k.Predict(x).first,k.Predict(x).second] for x in (-1,0,1,4,12)],p,rtol=2e-9,atol=2e-10)

@pytest.mark.parametrize('kind',[1,2,3])
def test_likelihood(kind):
    std=.6;mu=.08;scale=1.2
    if kind==1:
        th=[std,scale,mu,np.log(.3)]
        model=cm.CAR1(False,'test',v(T),v(Y),v(E))
        cov=lambda dt:std**2*np.exp(-.3*dt)
    else:
        ma=[1.,0.] if kind==2 else [1.,.4]
        roots=[-.8+1j,-.8-1j]
        th=[std,scale,mu,np.log(1.64),np.log(1.6)]+([] if kind==2 else [np.log(2.5)])
        model=(cm.CARp(False,'test',v(T),v(Y),v(E),2) if kind==2 else cm.CARMA(False,'test',v(T),v(Y),v(E),2,1))
        cov0=state_cov(roots,ma,1.)
        cov=lambda dt:std**2*cov0(dt)/cov0(0)
    expected=dense(cov,y=Y-mu,e=E*np.sqrt(scale))[3]
    actual=model.getLogDensity(v(th))-model.getLogPrior(v(th))
    assert np.isfinite(actual)
    assert_allclose(actual,expected,rtol=2e-10,atol=2e-10)

def parse(text):
    result={}
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0] in ['mean','var','pred','chain1','chain2','chain3','log1','log2','log3']:
            result.setdefault(fields[0],[]).append([float(x) for x in fields[1:]])
    return result

def run(q,thin=2,defaults=False):
    cm.set_seed(314159)
    args=(12,20,v(T),v(Y),v(E))
    if q==-1:
        init=v([.4,1.,.1,np.log(.3)])
        return cm.run_mcmc_car1(*args,init=init) if defaults else cm.run_mcmc_car1(*args,thin,init)
    return cm.run_mcmc_carma(*args,2,q,2) if defaults else cm.run_mcmc_carma(*args,2,q,2,False,thin)

def test_raw_cpp_reference():
    executable=os.environ['CARMA_REFERENCE_DRIVER']
    raw=subprocess.check_output([executable],text=True)
    ref=parse(raw)
    assert len(ref['chain1'])==12
    k=cm.KalmanFilter1(v(T),v(Y),v(E),.7,.3);k.Filter()
    assert_allclose(np.asarray(k.GetMean()),np.array(ref['mean'])[:,0],rtol=1e-14)
    assert_allclose(np.asarray(k.GetVar()),np.array(ref['var'])[:,0],rtol=1e-14)
    for q in (-1,0,1):
        model=run(q)
        assert_allclose(model.getSamples(),ref['chain'+str(q+2)],rtol=2e-13,atol=2e-13)
        assert_allclose(np.asarray(model.GetLogLikes()),np.array(ref['log'+str(q+2)])[:,0],rtol=2e-13)

@pytest.mark.parametrize('q',[-1,0,1])
def test_seed_defaults_thin_lifetime(q):
    a=run(q);gc.collect();x=np.array(a.getSamples())
    assert x.shape==(12,4 if q<0 else 5+q)
    assert np.isfinite(x).all()
    assert_allclose(x,run(q).getSamples(),rtol=0,atol=0)
    assert_allclose(run(q,defaults=True).getSamples(),run(q,thin=1).getSamples(),rtol=0,atol=0)
    assert not np.array_equal(x,np.array(run(q,thin=1).getSamples()))
    # Returned samples own their storage independently of their C++ model.
    samples=a.getSamples();del a;gc.collect();assert_allclose(samples,x)

@pytest.mark.parametrize('t,y,e',[(T[:2],Y,E),(T,Y[:-1],E),(T,Y,-E),(T[::-1],Y,E),(T,Y*float('nan'),E)])
def test_bad_data(t,y,e):
    with pytest.raises(ValueError):cm.run_mcmc_car1(2,2,v(t),v(y),v(e))

@pytest.mark.parametrize('n,b,thin',[(0,2,1),(2,-1,1),(2,2,0)])
def test_bad_chain(n,b,thin):
    with pytest.raises(ValueError):cm.run_mcmc_car1(n,b,v(T),v(Y),v(E),thin)

def test_vector_pair_and_constructor_defaults():
    x=v();x.extend([1,2]);x.append(3);x[0]=4;assert list(x)==[4,2,3]
    nested=cm.vecvecD();nested.append(x);assert list(nested[0])==[4,2,3]
    p=cm.pairD();p.first=2;p.second=3;assert(p.first,p.second)==(2,3)
    th=v([.6,1,.08,np.log(.3)])
    a=cm.CAR1(False,'test',v(T),v(Y),v(E));b=cm.CAR1(False,'test',v(T),v(Y),v(E),1.)
    assert a.getLogDensity(th)==b.getLogDensity(th)

def test_original_core_hashes():
    root=Path(__file__).resolve().parents[1]
    upstream=json.loads((root/'audit/upstream_hashes.json').read_text())
    changes=json.loads((root/'audit/phase2/core_changes.json').read_text())['changes']
    assert set(changes)=={'src/carpack.cpp','src/carmcmc.cpp'}
    for name,sha in upstream.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if name in changes:
            assert changes[name]['upstream_sha256']==sha
            assert actual==changes[name]['patched_sha256'] and actual!=sha
        else:
            assert actual==sha,name
    # Independent oracle snapshots must remain byte-for-byte upstream files.
    provenance=json.loads((root/'modern_tests/upstream_core/PROVENANCE.json').read_text())
    for name,sha in provenance['sha256'].items():
        assert sha==upstream[name]
        assert hashlib.sha256((root/'modern_tests/upstream_core'/name.removeprefix('src/')).read_bytes()).hexdigest()==sha

@pytest.mark.parametrize('q',[-1,0,1])
def test_public_python_model(q):
    cm.set_seed(42)
    model=cm.CarmaModel(T,Y,E,p=1 if q<0 else 2,q=max(q,0))
    init=v([.4,1.,.1,np.log(.3)]) if q<0 else None
    sample=model.run_mcmc(40,ntemperatures=2,init=init)
    assert model.mcmc_sample is sample
    assert np.isfinite(sample.get_samples('sigma')).all()

def test_bad_orders_and_initial_length():
    for p,q,w in [(1,0,2),(2,2,2),(2,-1,2),(2,0,0)]:
        with pytest.raises(ValueError):cm.run_mcmc_carma(2,2,v(T),v(Y),v(E),p,q,w)
    with pytest.raises(ValueError):cm.run_mcmc_car1(2,2,v(T),v(Y),v(E),1,v([1]))
    with pytest.raises(ValueError):cm.CAR1(False,'x',v(T),v(Y),v(E)).getLogDensity(v([1]))

@pytest.mark.parametrize('kind',[1,2])
def test_conditional_simulation(kind):
    cm.set_seed(718)
    if kind==1:
        k=cm.KalmanFilter1(v(T),v(Y),v(E),.7,.3)
        cov=lambda dt:.7/.6*np.exp(-.3*dt)
    else:
        k=cm.KalmanFilterp(v(T),v(Y),v(E),.7,cm.vecC([-1.,-.3]),v([1.,.4]))
        cov=state_cov([-1.,-.3],[1.,.4],.7)
    query=np.array([1.,3.,12.]);m=np.array(dense(cov,queries=query)[2])[:,0]
    K=np.array([[cov(abs(a-b)) for b in T] for a in T])+np.diag(E**2)
    cross=np.array([[cov(abs(a-b)) for b in T] for a in query])
    Q=np.array([[cov(abs(a-b)) for b in query] for a in query])-cross@np.linalg.solve(K,cross.T)
    draws=np.array([list(k.Simulate(v(query))) for _ in range(4000)])
    # Fixed-seed Monte Carlo checks with conservative 6-sigma sampling uncertainty.
    assert np.all(np.abs(draws.mean(0)-m)<6*np.sqrt(np.diag(Q)/len(draws)))
    uncertainty=np.sqrt((Q**2+np.outer(np.diag(Q),np.diag(Q)))/(len(draws)-1))
    assert np.all(np.abs(np.cov(draws.T)-Q)<6*uncertainty)
    assert_allclose([k.Predict(x).first for x in query],m,atol=2e-10)

def test_unsafe_filter_boundaries():
    for cls in (cm.KalmanFilter1,cm.KalmanFilterp):
        with pytest.raises(ValueError):cls(v(T),v(Y),v(E))
    with pytest.raises(ValueError):cm.KalmanFilterp(v(T),v(Y),v(E),.7,cm.vecC([-.3]),v([1.]))
    with pytest.raises(ValueError):cm.KalmanFilter1(v(T),v(Y),v(E),.7,.3).Predict(float('nan'))

def test_more_parameter_boundaries():
    with pytest.raises(ValueError):cm.CARp(False,'x',v(T),v(Y),v(E),1)
    with pytest.raises(ValueError):cm.CAR1(False,'x',v(T),v(Y),v(E),0.)
    with pytest.raises(ValueError):cm.KalmanFilter1(v(T),v(Y),v(E),float('inf'),.3)
    with pytest.raises(ValueError):cm.KalmanFilterp(v(T),v(Y),v(E),float('inf'),cm.vecC([-1.,-.3]),v([1.,0.]))
