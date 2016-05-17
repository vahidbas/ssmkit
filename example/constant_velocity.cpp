
#include "ssmpack/map/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"

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


  std::vector<typename decltype(joint_process)::TRandomVARs> v;

  random::setRandomSeed();
  joint_process.initialize();
  joint_process.random_n(v, 100);

  std::for_each(v.begin(), v.end(), [](auto &p) {
    std::cout << std::get<0>(p) << std::endl
              << std::get<1>(p) << "-----------" << std::endl;
  });
}
