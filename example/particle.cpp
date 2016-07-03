#include "ssmkit/filter/particle.hpp"
#include "ssmkit/filter/resampler/systematic.hpp"
#include "ssmkit/filter/resampler/criterion/ess.hpp"
#include "ssmkit/map/linear_gaussian.hpp"
#include "ssmkit/distribution/gaussian.hpp"
#include "ssmkit/distribution/conditional.hpp"
#include "ssmkit/process/markov.hpp"
#include "ssmkit/process/memoryless.hpp"
#include "ssmkit/process/hierarchical.hpp"

#include <iostream>

using namespace ssmkit;

int main() {
  double delta = 0.1; // sample time
  arma::mat dynamic_matrix{
      {1, 0, delta, 0}, {0, 1, 0, delta}, {0, 0, 1, 0}, {0, 0, 0, 1}};

  arma::mat dynamic_noise{
      {0.1, 0, 0, 0}, {0, 0.1, 0, 0}, {0, 0, 0.1, 0}, {0, 0, 0, 0.1}};

  arma::mat measurement_matrix{{1, 0, 0, 0}, {0, 1, 0, 0}};
  arma::mat measurement_noise{{0.1, 0}, {0, 0.1}};

  auto dynamic_model = map::LinearGaussian(dynamic_matrix, dynamic_noise);
  auto measurement_model =
      map::LinearGaussian(measurement_matrix, measurement_noise);

  auto dynamic_cpdf =
      distribution::makeConditional(distribution::Gaussian(4), dynamic_model);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian(2), measurement_model);

  auto state_process =
      process::makeMarkov(dynamic_cpdf, distribution::Gaussian(4));
  auto measurement_process = process::makeMemoryless(measurement_cpdf);

  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);

  auto pfilter = filter::makeParticle(
      joint_process,
      filter::resampler::makeSystematic(filter::resampler::criterion::ESS(40)),
      50);

  random::setRandomSeed();

  pfilter.initialize();

  // std::cout << pfilter.getWeights();
  // std::vector<typename decltype(joint_process)::TRandomVAR> v;
  joint_process.initialize();
  int n = 20;
  auto v = joint_process.random_n(n);
  std::vector<typename std::tuple_element<1, decltype(v)::value_type>::type> m(
      n);
  int count = 0;
  std::generate_n(m.begin(), m.size(),
                  [&count, &v]() { return std::get<1>(v[count++]); });
  std::for_each(m.begin(), m.end(),
                [](auto &e) { std::cout << e << "------" << std::endl; });
  auto o = pfilter.filter(m);
  return 0;
}
