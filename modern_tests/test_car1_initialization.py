"""Initialization correctness; movement here is a regression check, NOT convergence."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_science import cm,v,T,Y,E,dense


def series(kind):
    if kind=='reported':return T,Y,E
    if kind=='seasonal':
        t=np.r_[np.linspace(0,17,18),np.linspace(145,173,14),np.linspace(360,392,16)]
        t=t+0.08*np.sin(np.arange(len(t)))
        return t,19+.31*np.sin(t/13)+.15*np.cos(t/4),.05+.02*(1+np.sin(t))
    t=np.array([0,.001,.003,.025,.11,.5,1.,2.3,12.,25.,91.,225.,700.,730.])
    return t,.2*np.sin(t*1.3)+.3*np.cos(np.arange(len(t))),.06+.01*np.arange(len(t))/len(t)


@pytest.mark.parametrize('kind',['reported','seasonal','multiscale'])
@pytest.mark.parametrize('seed',[0,1,2,42,82,314159,271828,4294967295])
def test_automatic_initialization_finite_legal_moving(kind,seed):
    t,y,e=series(kind);cm.set_seed(seed)
    model=cm.run_mcmc_car1(64,64,v(t),v(y),v(e),2)
    x=np.asarray(model.getSamples());post=np.asarray(model.GetLogLikes())
    assert np.isfinite(post).all(),('nonfinite cached posterior',kind,seed)
    assert np.isfinite(x).all()
    assert (x[:,0]>=0).all() and (x[:,0]<=10*np.std(y)).all()
    assert ((x[:,1]>=.5)&(x[:,1]<=2)).all()
    omega=np.exp(x[:,3])
    assert (omega>=1/np.ptp(t)).all() and (omega<=1/np.min(np.diff(t))).all()
    assert np.any(np.diff(x,axis=0)!=0),('frozen chain',kind,seed)
    assert_allclose([model.getLogDensity(v(row)) for row in x],post,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('init',[
    [.4,1.,.1,-100.], [.4,1.,.1,100.], [-.4,1.,.1,-1.],
    [100.,1.,.1,-1.], [.4,0.,.1,-1.], [.4,3.,.1,-1.],
    [.4,1.,float('nan'),-1.], [.4,1.,.1,float('inf')], [1.],
])
def test_invalid_explicit_initialization_is_rejected(init):
    with pytest.raises(ValueError):
        cm.run_mcmc_car1(4,2,v(T),v(Y),v(E),1,v(init))


@pytest.mark.parametrize('t,y,e',[
    (T,np.ones(len(T)),E),
    (T[:2],Y[:2],E[:2]),
    (np.array([]),np.array([]),np.array([])),
    (T,Y,np.zeros(len(T))),
    (np.zeros(len(T)),Y,E),
])
def test_degenerate_input_is_rejected(t,y,e):
    with pytest.raises(ValueError):cm.run_mcmc_car1(4,2,v(t),v(y),v(e))

# Native probes inspect the true starting point before any MCMC transition.
import os,subprocess

def probe_input(t,y,e,seed,nsamples=64,burnin=64,thin=2,init=None):
    data=f'{len(t)} {seed} {nsamples} {burnin} {thin}\n'
    data+=''.join(f'{a:.17g} {b:.17g} {c:.17g}\n' for a,b,c in zip(t,y,e))
    if init is not None:data+=' '.join(f'{x:.17g}' for x in init)+'\n'
    return data

def parse_probe(text):
    result={}
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0] in ('initial','bounds','legal','density','finite_candidate','after','proposal_diag','sample','posterior'):
            result.setdefault(fields[0],[]).append([float(x) for x in fields[1:]])
    return result

@pytest.mark.parametrize('kind',['reported','seasonal','multiscale'])
@pytest.mark.parametrize('seed',[0,1,2,42,82,314159,271828,4294967295])
def test_native_start_legal_finite_and_cache_consistent(kind,seed):
    t,y,e=series(kind)
    raw=subprocess.check_output([os.environ['CARMA_CURRENT_PROBE'],'inspect'],input=probe_input(t,y,e,seed),text=True,timeout=10)
    values=parse_probe(raw);theta=np.array(values['initial'][0])
    assert values['legal']==[[1.]]
    direct,cached=values['density'][0]
    assert np.isfinite([direct,cached]).all() and direct==cached
    cov=lambda dt:theta[0]**2*np.exp(-np.exp(theta[3])*dt)
    ll=dense(cov,t=t,y=y-theta[2],e=e*np.sqrt(theta[1]))[3]
    prior=-25/theta[1]-26*np.log(theta[1])
    assert_allclose(direct,ll+prior,rtol=2e-10,atol=2e-10)
    assert values['after'][0][0]>0
    assert np.isfinite(values['after'][0][1])

@pytest.mark.parametrize('seed',[0,42,82,314159,271828])
@pytest.mark.parametrize('thin',[1,3])
def test_explicit_path_bitwise_original_oracle(seed,thin):
    init=[.4,1.,.1,np.log(.3)]
    data=probe_input(T,Y,E,seed,24,32,thin,init)
    reference=parse_probe(subprocess.check_output([os.environ['CARMA_UPSTREAM_PROBE'],'explicit'],input=data,text=True,timeout=10))
    cm.set_seed(seed)
    model=cm.run_mcmc_car1(24,32,v(T),v(Y),v(E),thin,v(init))
    assert_allclose(model.getSamples(),reference['sample'],rtol=0,atol=0)
    assert_allclose(np.asarray(model.GetLogLikes()),np.array(reference['posterior'])[:,0],rtol=0,atol=0)

@pytest.mark.parametrize('seed',[42,314159,271828])
def test_auto_seed_reproduction_and_saved_posterior_vs_independent_ou(seed):
    cm.set_seed(seed);a=cm.run_mcmc_car1(16,32,v(T),v(Y),v(E),2)
    cm.set_seed(seed);b=cm.run_mcmc_car1(16,32,v(T),v(Y),v(E),2)
    assert_allclose(a.getSamples(),b.getSamples(),rtol=0,atol=0)
    for theta,post in zip(a.getSamples(),a.GetLogLikes()):
        theta=np.array(theta)
        cov=lambda dt:theta[0]**2*np.exp(-np.exp(theta[3])*dt)
        ll=dense(cov,y=Y-theta[2],e=E*np.sqrt(theta[1]))[3]
        assert_allclose(a.getLogPrior(v(theta)),-25/theta[1]-26*np.log(theta[1]),rtol=2e-13)
        assert_allclose(post,ll-25/theta[1]-26*np.log(theta[1]),rtol=2e-11,atol=2e-11)


def test_automatic_retry_budget_raises_instead_of_hanging():
    # Test-only narrow prior makes all 128 draws invalid for this fixed seed.
    r=subprocess.run([os.environ['CARMA_CURRENT_PROBE'],'exhaust'],input=probe_input(T,Y,E,314159),text=True,capture_output=True,timeout=10)
    assert r.returncode==3
    assert 'after 128 attempts' in r.stderr

@pytest.mark.parametrize('t,y,e',[
    (T,np.full(len(T),1e100)+np.arange(len(T)),E),
    (T,Y*1e-200,E),
    (T,Y*1e200,E),
    (np.array([0,1e-320,1.,2.,3.,4.,5.]),Y,E),
])
def test_unrepresentable_variance_or_frequency_is_rejected(t,y,e):
    with pytest.raises(ValueError):cm.run_mcmc_car1(4,2,v(t),v(y),v(e))

@pytest.mark.parametrize('n,b,thin',[(0,2,1),(2,-1,1),(2,2,0),(2,2,2147483647)])
def test_native_boundary_validation(n,b,thin):
    r=subprocess.run([os.environ['CARMA_CURRENT_PROBE'],'auto'],input=probe_input(T,Y,E,314159,n,b,thin),text=True,capture_output=True,timeout=10)
    assert r.returncode==3 and 'ERROR:' in r.stderr

@pytest.mark.parametrize('kind',['constant','short','bad-init'])
def test_native_degenerate_and_explicit_errors(kind):
    t,y,e=T,Y,E;init=None;mode='auto'
    if kind=='constant':y=np.ones(len(T))
    if kind=='short':t,y,e=T[:2],Y[:2],E[:2]
    if kind=='bad-init':mode='explicit';init=[.4,1.,.1,-100.]
    r=subprocess.run([os.environ['CARMA_CURRENT_PROBE'],mode],input=probe_input(t,y,e,314159,init=init),text=True,capture_output=True,timeout=10)
    assert r.returncode==3 and 'ERROR:' in r.stderr


def test_finite_explicit_parameters_with_nonfinite_density_are_rejected():
    # theta is finite and within the original (closed) bounds. sigma=0 and
    # underflowing error variances nevertheless give a nonfinite density.
    with pytest.raises(ValueError,match='finite log density'):
        cm.run_mcmc_car1(4,2,v(T),v(Y),v(np.full(len(T),1e-300)),1,v([0.,1.,0.,np.log(.3)]))
