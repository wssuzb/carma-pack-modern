// Test-only direct C++ evidence. No Python/binding or modified transition kernel.
#include "carmcmc.hpp"
#include "random.hpp"
#include <iomanip>
#include <iostream>
#include <string>

class InspectCAR1 : public CAR1 {
public:
  using CAR1::CAR1;
  double low() const { return min_freq_; }
  double high() const { return max_freq_; }
  double maxstd() const { return max_stdev_; }
};

int main(int argc, char **argv) {
  try {
    std::string mode = argc > 1 ? argv[1] : "inspect";
    int n, nsamples, burnin, thin;
    unsigned long seed;
    if (!(std::cin >> n >> seed >> nsamples >> burnin >> thin)) return 2;
    std::vector<double> t(n), y(n), e(n);
    for (int i=0; i<n; ++i) std::cin >> t[i] >> y[i] >> e[i];
    RandomGenerator r; r.SetSeed(seed); arma::arma_rng::set_seed(seed);
    std::cout << std::setprecision(17);
    if (mode=="explicit" || mode=="auto") {
      std::vector<double> init;
      if (mode=="explicit") {init.resize(4);for(auto& x:init)std::cin >> x;}
      auto model=RunCar1Sampler(nsamples,burnin,t,y,e,thin,init);
      for(auto row:model->getSamples()) {std::cout<<"sample";for(auto x:row)std::cout<<" "<<x;std::cout<<"\n";}
      for(auto x:model->GetLogLikes())std::cout<<"posterior "<<x<<"\n";
      return 0;
    }
    InspectCAR1 model(false,"probe",t,y,e);
    double mean=std::accumulate(y.begin(),y.end(),0.)/n;
    double var=std::inner_product(y.begin(),y.end(),y.begin(),0.)/n-mean*mean;
    model.SetPrior(mode=="exhaust" ? 1e-100 : 10*std::sqrt(var));
    auto theta=model.StartingValue();
    std::cout<<"initial";for(auto x:theta)std::cout<<" "<<x;std::cout<<"\n";
    std::cout<<"bounds "<<model.low()<<" "<<model.high()<<" "<<model.maxstd()<<"\n";
    std::cout<<"legal "<<model.CheckPriorBounds(theta)<<"\n";
    // Sampler::Run operates through Parameter<arma::vec>*, whose Save is the
    // actual virtual by-value overload. CARMA_Base::Save(arma::vec&) hides it.
    Parameter<arma::vec>& par=model;par.Save(theta);
    double direct=model.LogDensity(theta);
    std::cout<<"density "<<direct<<" "<<model.GetLogDensity()<<"\n";
    arma::mat cov=arma::eye<arma::mat>(4,4)*.0001;
    cov(0,0)=2*var*var/n;cov(2,2)=var/n;
    StudentProposal proposal(8.,1.);
    AdaptiveMetro step(par,proposal,cov,.25,burnin);
    arma::vec good=theta;
    good(0)=std::sqrt(var);good(1)=1.;good(2)=mean;
    good(3)=.5*(std::log(model.low())+std::log(model.high()));
    double good_density=model.LogDensity(good);
    bool accepted=step.Accept(good,theta);
    std::cout<<"finite_candidate "<<good_density<<" "<<accepted<<" "<<step.GetMetroRatio()<<"\n";
    // Accept does not Save; run the unchanged update on the original state.
    for(int i=0;i<burnin;++i)step.DoStep();
    std::cout<<"after "<<arma::norm(model.Value()-theta)<<" "<<model.GetLogDensity()<<"\n";
    std::cout<<"proposal_diag";for(auto x:arma::vec(step.GetCovariance().diag()))std::cout<<" "<<x;std::cout<<"\n";
    return 0;
  } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<"\n";return 3;}
}
