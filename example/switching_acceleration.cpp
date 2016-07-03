#include "ssmkit/map/linear_gaussian.hpp"
#include "ssmkit/map/switching_additive_linear_gaussian.hpp"
#include "ssmkit/map/transition_matrix.hpp"
#include "ssmkit/distribution/gaussian.hpp"
#include "ssmkit/distribution/categorical.hpp"
#include "ssmkit/distribution/conditional.hpp"
#include "ssmkit/process/markov.hpp"
#include "ssmkit/process/memoryless.hpp"
#include "ssmkit/process/hierarchical.hpp"

#include <iostream>

using namespace ssmkit;

int main() {
  double delta = 1; // sample time

  // switching process
  arma::mat transition_matrix{
      {0.8, 0.1, 0.1}, {0.1, 0.8, 0.1}, {0.1, 0.1, 0.8}};
  auto switching_model = map::TransitionMatrix(transition_matrix);
  auto switching_cpdf = distribution::makeConditional(
      distribution::Categorical(), switching_model);
  auto switching_process = process::makeMarkov(
      switching_cpdf, distribution::Categorical({0.4, 0.3, 0.3}));

  // dynamic process
  arma::mat dynamic_matrix{{1, delta}, {0, 1}};
  arma::mat dynamic_noise{{0.1, 0}, {0, 0.1}};
  arma::mat accelerations{
      {0, delta * delta / 2, -delta * delta / 2}, {0, delta, -delta}};
  auto state_model = map::SwitchingAdditiveLinearGaussian(
      dynamic_matrix, dynamic_noise, accelerations);
  auto state_cpdf =
      distribution::makeConditional(distribution::Gaussian(2), state_model);
  auto state_process =
      process::makeMarkov(state_cpdf, distribution::Gaussian(2));

  // measurement process
  arma::mat measurement_matrix{1, 0};
  arma::mat measurement_noise{0.1};
  auto measurement_model =
      map::LinearGaussian(measurement_matrix, measurement_noise);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian(1), measurement_model);
  auto measurement_process = process::makeMemoryless(measurement_cpdf);

  // joint process
  auto joint_process = process::makeHierarchical(
      switching_process, state_process, measurement_process);

  // simulation
  joint_process.initialize();
  auto v = joint_process.random_n(100);

  std::for_each(v.begin(), v.end(), [](auto &p) {
    std::cout << std::get<0>(p) << std::endl
              << std::get<1>(p) << std::endl
              << std::get<2>(p) << "-----------" << std::endl;
  });
}
