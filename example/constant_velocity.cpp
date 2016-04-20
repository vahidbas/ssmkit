
#include "ssmpack/model/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"

using namespace ssmpack;
int main ()
{

  double delta = 0.1; // sample time
  arma::mat::fixed<4, 4> dynamic_matrix{
      {1, 0, delta, 0}, {0, 1, 0, delta}, {0, 0, 1, 0}, {0, 0, 0, 1}};

  arma::mat::fixed<4, 4> dynamic_noise{
      {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

  arma::mat::fixed<2, 4> measurement_matrix{{1, 0, 0, 0}, {0, 1, 0, 0}};
  arma::mat::fixed<2, 2> measurement_noise{{1, 0}, {0, 1}};

  auto dynamic_model = model::makeLinearGaussian(dynamic_matrix, dynamic_noise);
  auto measurement_model =
      model::makeLinearGaussian(measurement_matrix, measurement_noise);

  auto dynamic_cpdf =
      distribution::makeConditional(distribution::Gaussian<4>(), dynamic_model);
  auto measurement_cpdf = distribution::makeConditional(
      distribution::Gaussian<2>(), measurement_model);

  auto state_process =
      process::makeMarkov(dynamic_cpdf, distribution::Gaussian<4>());
  auto measurement_process =
      process::makeMemoryless(measurement_cpdf);

  auto joint_process =
      process::makeHierarchical(state_process, measurement_process);

  joint_process.random();
}
