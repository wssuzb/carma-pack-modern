#include "include/carmcmc.hpp"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;
using VD = std::vector<double>;
using VC = std::vector<std::complex<double>>;
using VVD = std::vector<VD>;
using PD = std::pair<double, double>;
PYBIND11_MAKE_OPAQUE(PD);
PYBIND11_MAKE_OPAQUE(VD);
PYBIND11_MAKE_OPAQUE(VC);
PYBIND11_MAKE_OPAQUE(VVD);
static void data(const VD &t, const VD &y, const VD &e) {
  if (t.size() < 3 || t.size() != y.size() || t.size() != e.size())
    throw py::value_error("equal data lengths >=3 required");
  for (size_t i = 0; i < t.size(); ++i)
    if (!std::isfinite(t[i]) || !std::isfinite(y[i]) || !std::isfinite(e[i]) ||
        e[i] <= 0 || (i && t[i] <= t[i - 1]))
      throw py::value_error(
          "finite, strictly increasing times and positive errors required");
}
static void chain(int n, int b, int thin, const VD &t, const VD &y,
                  const VD &e) {
  data(t, y, e);
  if (n <= 0 || b < 0 || thin <= 0)
    throw py::value_error("invalid sample_size, burnin or thin");
  if (*std::max_element(y.begin(), y.end()) ==
      *std::min_element(y.begin(), y.end()))
    throw py::value_error("nonconstant data required");
}
template <class C> void methods(C &c) {
  using T = typename C::type;
  auto check = [](T &self, const VD &theta) {
    if (theta.size() != self.Value().n_elem)
      throw py::value_error("wrong parameter length");
    for (double x : theta)
      if (!std::isfinite(x))
        throw py::value_error("finite parameters required");
  };
  c.def("getLogPrior",
        [check](T &self, VD theta) {
          check(self, theta);
          return self.getLogPrior(theta);
        })
      .def("getLogDensity",
           [check](T &self, VD theta) {
             check(self, theta);
             return self.getLogDensity(theta);
           })
      .def("getSamples", &T::getSamples)
      .def("GetLogLikes", &T::GetLogLikes);
}
PYBIND11_MODULE(_carmcmc, m) {
  py::bind_vector<VD>(m, "vecD");
  py::bind_vector<VC>(m, "vecC");
  py::bind_vector<VVD>(m, "vecvecD");
  py::class_<std::pair<double, double>>(m, "pairD")
      .def(py::init<>())
      .def_readwrite("first", &std::pair<double, double>::first)
      .def_readwrite("second", &std::pair<double, double>::second);
  py::class_<CARMA_Base<double>, std::shared_ptr<CARMA_Base<double>>>(
      m, "CARMA_Base_double");
  py::class_<CARMA_Base<arma::cx_vec>,
             std::shared_ptr<CARMA_Base<arma::cx_vec>>>(m, "CARMA_Base_arma");
  auto a =
      py::class_<CAR1, CARMA_Base<double>, std::shared_ptr<CAR1>>(m, "CAR1");
  a.def(py::init([](bool tr, std::string name, VD t, VD y, VD e, double temp) {
          data(t, y, e);
          if (!std::isfinite(temp) || temp <= 0)
            throw py::value_error("positive finite temperature required");
          return std::make_shared<CAR1>(tr, name, t, y, e, temp);
        }),
        py::arg("track"), py::arg("name"), py::arg("time"), py::arg("y"),
        py::arg("yerr"), py::arg("temperature") = 1.0);
  methods(a);
  auto b = py::class_<CARp, CARMA_Base<arma::cx_vec>, std::shared_ptr<CARp>>(
      m, "CARp");
  b.def(py::init([](bool tr, std::string name, VD t, VD y, VD e, int p,
                    double temp) {
          data(t, y, e);
          if (p < 2)
            throw py::value_error("p >=2 required; use CAR1 for p=1");
          if (!std::isfinite(temp) || temp <= 0)
            throw py::value_error("positive finite temperature required");
          return std::make_shared<CARp>(tr, name, t, y, e, p, temp);
        }),
        py::arg("track"), py::arg("name"), py::arg("time"), py::arg("y"),
        py::arg("yerr"), py::arg("p"), py::arg("temperature") = 1.0)
      .def("SetMLE", &CARp::SetMLE);
  methods(b);
  auto c = py::class_<CARMA, CARp, std::shared_ptr<CARMA>>(m, "CARMA");
  c.def(py::init([](bool tr, std::string name, VD t, VD y, VD e, int p, int q,
                    double temp) {
          data(t, y, e);
          if (p < 2 || q < 1 || q >= p)
            throw py::value_error("0<q<p required; use CARp for q=0");
          if (!std::isfinite(temp) || temp <= 0)
            throw py::value_error("positive finite temperature required");
          return std::make_shared<CARMA>(tr, name, t, y, e, p, q, temp);
        }),
        py::arg("track"), py::arg("name"), py::arg("time"), py::arg("y"),
        py::arg("yerr"), py::arg("p"), py::arg("q"),
        py::arg("temperature") = 1.0)
      .def("SetMLE", &CARMA::SetMLE);
  methods(c);
  m.def("set_seed", [](unsigned long seed) {
    RandomGenerator r;
    r.SetSeed(seed);
    arma::arma_rng::set_seed(seed);
  });
  m.def(
      "run_mcmc_car1",
      [](int n, int b, VD t, VD y, VD e, int thin, VD init) {
        chain(n, b, thin, t, y, e);
        if (init.size() && init.size() != 4)
          throw py::value_error("init length must be 4");
        return RunCar1Sampler(n, b, t, y, e, thin, init);
      },
      py::arg("sample_size"), py::arg("burnin"), py::arg("time"), py::arg("y"),
      py::arg("yerr"), py::arg("thin") = 1, py::arg("init") = VD{});
  m.def(
      "run_mcmc_carma",
      [](int n, int b, VD t, VD y, VD e, int p, int q, int w, bool z, int thin,
         VD init) {
        chain(n, b, thin, t, y, e);
        if (p < 2 || q < 0 || q >= p || w < 1)
          throw py::value_error("require p>=2, 0<=q<p, nwalkers>=1");
        if (init.size() && init.size() != size_t(3 + p + (z ? 0 : q)))
          throw py::value_error("wrong init length");
        return RunCarmaSampler(n, b, t, y, e, p, q, w, z, thin, init);
      },
      py::arg("sample_size"), py::arg("burnin"), py::arg("time"), py::arg("y"),
      py::arg("yerr"), py::arg("p"), py::arg("q"), py::arg("nwalkers"),
      py::arg("do_zcarma") = false, py::arg("thin") = 1,
      py::arg("init") = VD{});
  py::class_<KalmanFilter<double>, std::shared_ptr<KalmanFilter<double>>>(
      m, "KalmanFilter_double");
  py::class_<KalmanFilter<arma::cx_vec>,
             std::shared_ptr<KalmanFilter<arma::cx_vec>>>(m,
                                                          "KalmanFilter_arma");
  py::class_<KalmanFilter1, KalmanFilter<double>,
             std::shared_ptr<KalmanFilter1>>(m, "KalmanFilter1")
      .def(py::init([](VD t, VD y, VD e) -> std::shared_ptr<KalmanFilter1> {
        data(t, y, e);
        throw py::value_error(
            "explicit process parameters required; upstream data-only "
            "constructor leaves them uninitialized");
      }))
      .def(py::init([](VD t, VD y, VD e, double s, double o) {
        data(t, y, e);
        if (!(std::isfinite(s) && std::isfinite(o) && s > 0 && o > 0))
          throw py::value_error("positive sigsqr and omega required");
        return std::make_shared<KalmanFilter1>(t, y, e, s, o);
      }))
      .def("Filter", &KalmanFilter1::Filter)
      .def("Predict",
           [](KalmanFilter1 &self, double t) {
             if (!std::isfinite(t))
               throw py::value_error("finite prediction time required");
             return self.Predict(t);
           })
      .def("Simulate",
           [](KalmanFilter1 &self, VD t) {
             for (double x : t)
               if (!std::isfinite(x))
                 throw py::value_error("finite simulation times required");
             return self.Simulate(t);
           })
      .def("GetMean", &KalmanFilter1::GetMeanSvec)
      .def("GetVar", &KalmanFilter1::GetVarSvec);
  py::class_<KalmanFilterp, KalmanFilter<arma::cx_vec>,
             std::shared_ptr<KalmanFilterp>>(m, "KalmanFilterp")
      .def(py::init([](VD t, VD y, VD e) -> std::shared_ptr<KalmanFilterp> {
        data(t, y, e);
        throw py::value_error(
            "explicit process parameters required; upstream data-only "
            "constructor leaves them uninitialized");
      }))
      .def(py::init([](VD t, VD y, VD e, double s, VC o, VD ma) {
        data(t, y, e);
        if (!(std::isfinite(s) && s > 0) || o.size() < 2 ||
            ma.size() != o.size())
          throw py::value_error(
              "positive sigsqr, p>=2 and p MA coefficients required");
        for (auto r : o)
          if (!std::isfinite(r.real()) || !std::isfinite(r.imag()) ||
              r.real() >= 0)
            throw py::value_error("stable finite roots required");
        return std::make_shared<KalmanFilterp>(t, y, e, s, o, ma);
      }))
      .def("Filter", &KalmanFilterp::Filter)
      .def("Predict",
           [](KalmanFilterp &self, double t) {
             if (!std::isfinite(t))
               throw py::value_error("finite prediction time required");
             return self.Predict(t);
           })
      .def("Simulate",
           [](KalmanFilterp &self, VD t) {
             for (double x : t)
               if (!std::isfinite(x))
                 throw py::value_error("finite simulation times required");
             return self.Simulate(t);
           })
      .def("GetMean", &KalmanFilterp::GetMeanSvec)
      .def("GetVar", &KalmanFilterp::GetVarSvec);
}
