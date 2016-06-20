
#include "ssmpack/map/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"
#include "ssmpack/filter/kalman.hpp"

#include <iostream>

using namespace ssmpack;

int main ()
{

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
  auto measurement_process =
      process::makeMemoryless(measurement_cpdf);

  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);

  auto kalman = filter::makeKalman(joint_process);

  // std::vector<typename decltype(joint_process)::TRandomVAR> v;

  random::setRandomSeed();
  joint_process.initialize();
  auto v = joint_process.random_n(100);

  std::vector<typename std::tuple_element<1,decltype(v)::value_type>::type> m(100);
  int count=0;
  std::generate_n(m.begin(), 100,
                  [&count, &v]() { return std::get<1>(v[count++]); });
  auto o = kalman.filter(m);

}
