#include "carmcmc.hpp"
#include "random.hpp"
#include <iomanip>
int main() {
  std::cout << std::setprecision(17);
  std::vector<double> t{0, 0.4, 1.7, 2.1, 4.8, 7, 10},
      y{0.2, -0.1, 0.5, 0.3, -0.4, 0.1, 0.2},
      e{0.1, 0.15, 0.1, 0.2, 0.12, 0.1, 0.15};
  KalmanFilter1 k(t, y, e, 0.7, 0.3);
  k.Filter();
  for (auto x : k.GetMeanSvec())
    std::cout << "mean " << x << "\n";
  for (auto x : k.GetVarSvec())
    std::cout << "var " << x << "\n";
  for (double x : {-1., 0., 1., 4., 12.}) {
    auto v = k.Predict(x);
    std::cout << "pred " << v.first << " " << v.second << "\n";
  }
  for (int q : {-1, 0, 1}) {
    RandomGenerator r;
    r.SetSeed(314159);
    arma::arma_rng::set_seed(314159);
    auto m = q < 0
                 ? std::static_pointer_cast<CARMA_Base<double>>(RunCar1Sampler(
                       12, 20, t, y, e, 2, {0.4, 1.0, 0.1, std::log(0.3)}))
                 : nullptr;
    if (q < 0) {
      for (auto row : m->getSamples()) {
        std::cout << "chain1";
        for (auto x : row)
          std::cout << " " << x;
        std::cout << "\n";
      }
      for (auto x : m->GetLogLikes())
        std::cout << "log1 " << x << "\n";
    } else {
      auto n = RunCarmaSampler(12, 20, t, y, e, 2, q, 2, false, 2);
      for (auto row : n->getSamples()) {
        std::cout << "chain" << q + 2;
        for (auto x : row)
          std::cout << " " << x;
        std::cout << "\n";
      }
      for (auto x : n->GetLogLikes())
        std::cout << "log" << q + 2 << " " << x << "\n";
    }
  }
}
