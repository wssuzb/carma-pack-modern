import pytest
import numpy as np
from test_science import cm,T,Y,E,v

@pytest.mark.parametrize("p", [2, 3, 4, 6])
def test_car6_prediction_short_ma_matches_padded_native(p):
    sample=cm.CarmaSample.__new__(cm.CarmaSample)
    roots=-np.arange(1,p+1)*.1
    sample.time=T;sample.y=Y;sample.ysig=E;sample.p=p;sample.q=0
    sample._samples={'logpost':np.array([[0.]]),'sigma':np.array([[.7]]),
      'mu':np.array([[.1]]),'ar_roots':roots[None,:],'ma_coefs':np.array([[1.]])}
    k,mu=sample.makeKalmanFilter('map');k.Filter()
    ref=cm.KalmanFilterp(v(T),v(Y-.1),v(E),.7**2,cm.vecC(roots),v([1]+[0]*(p-1)));ref.Filter()
    for x in [-1.,1.,12.]:
        a=k.Predict(x);b=ref.Predict(x)
        np.testing.assert_allclose([a.first,a.second],[b.first,b.second],rtol=1e-12)
