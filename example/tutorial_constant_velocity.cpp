#include "ssmpack/map/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"
#include "ssmpack/process/hierarchical.hpp"
#include "ssmpack/filter/kalman.hpp"

#include <algorithm>
#include <numeric>
#include <iostream>


using namespace ssmpack;

int main ()
{
  // state dimensions
  int state_dim = 4;
  // measurement dimensions
  int meas_dim = 2;
  // sample time
  double delta = 0.1;
  // state transition matrix
  arma::mat F{{1, 0, delta,     0},
              {0, 1,     0, delta},
              {0, 0,     1,     0},
              {0, 0,     0,     1}};
  // dynamic noise covariance matrix
  arma::mat Q{{1, 0, 0, 0},
              {0, 1, 0, 0},
              {0, 0, 1, 0},
              {0, 0, 0, 1}};
  // measurement matrix
  arma::mat H{{1, 0, 0, 0},
              {0, 1, 0, 0}};
  // measurement noise covariance matrix
  arma::mat R{{1, 0},
              {0, 1}};
  // initial state pdf mean vector
  arma::vec x0{0, 0, 0, 0};
  // initial state pdf covariance matrix
  arma::mat P0{{1, 0, 0, 0},
               {0, 1, 0, 0},
               {0, 0, 1, 0},
               {0, 0, 0, 1}};
  // initial state pdf
  ssmpack::distribution::Gaussian initial_pdf(x0, P0);
  // state cpdf
  auto state_cpdf = ssmpack::distribution::makeConditional(
      ssmpack::distribution::Gaussian(state_dim),
      ssmpack::map::LinearGaussian(F, Q));
  // measurement cpdf
  auto meas_cpdf = ssmpack::distribution::makeConditional(
      ssmpack::distribution::Gaussian(meas_dim),
      ssmpack::map::LinearGaussian(H, R));
  
  // state process
  auto state_proc = ssmpack::process::makeMarkov(state_cpdf, initial_pdf);
  // measurement process
  auto meas_proc = ssmpack::process::makeMemoryless(meas_cpdf);

  // the hierarchical state space model
  auto ssm_proc = ssmpack::process::makeHierarchical(state_proc, meas_proc);

  // all in one statement!
//  auto ssm_proc =
//    makeHierarchical(
//      makeMarkov(
//        makeConditional(Gaussian(state_dim), LinearGaussian(F, Q)),
//        Gaussian(x0, P0)),
//      makeMemoryless(
//          makeConditional(Gaussian(meas_dim), LinearGaussian(H, R))));

  // initialize the process
  ssm_proc.initialize();
  // simulate data
  size_t n = 100;
  auto data = ssm_proc.random_n(n);

  // seperate state (x) and measurement (z) sequences
  std::vector<typename std::tuple_element<0,decltype(data)::value_type>::type> x_seq(n);
  std::vector<typename std::tuple_element<1,decltype(data)::value_type>::type> z_seq(n);
  std::transform(data.begin(), data.end(), x_seq.begin(),
                 [](const auto &v) { return std::get<0>(v); });
  std::transform(data.begin(), data.end(), z_seq.begin(),
                 [](const auto &v) { return std::get<1>(v); });


  // make kalman filter
  auto kalman = ssmpack::filter::makeKalman(ssm_proc);
  // apply Kalman filter on the simulated measurement to estimate x sequence
  auto x_seq_est_kalman = kalman.filter(z_seq);
  
  // calculate mean squared error of the estimation
  auto mse = std::inner_product(
      x_seq.begin(), x_seq.end(), x_seq_est_kalman.begin() + 1,
      static_cast<arma::vec>(arma::zeros<arma::vec>(state_dim)),
      std::plus<arma::vec>(), [n](const auto &a, const auto &b) {
        return static_cast<arma::vec>(arma::square(a - std::get<0>(b)) / n);
      });

  std::cout << "Kalman MSE = \n" << mse << std::endl;
}



