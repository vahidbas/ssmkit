#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "ssmpack/distribution/gaussian.hpp"

using namespace ssmpack;

BOOST_AUTO_TEST_SUITE(distribution_gausian);

constexpr int mc_n = 500000;

BOOST_AUTO_TEST_CASE(mc_test_random_default_pdf)
{
  constexpr int dimension = 8;
  distribution::Gaussian<dimension> pdf;


  std::vector<arma::vec> rvs(mc_n);

  random::setRandomSeed();

  arma::mat samples(dimension,mc_n);
  samples.each_col([&pdf](arma::vec &col){col = pdf.random();});
  auto mc_mean = arma::sum(samples,1)/mc_n;
  auto mc_covariance = arma::cov(samples.t());
  BOOST_CHECK(arma::approx_equal(mc_mean, arma::zeros<arma::vec>(dimension),
                                 "absdiff", 0.01));
  BOOST_CHECK(arma::approx_equal(mc_covariance,
                                 arma::eye<arma::mat>(dimension, dimension),
                                 "absdiff", 0.01));
}

BOOST_AUTO_TEST_CASE(mc_test_random_arbitary_pdf)
{
  constexpr int dimension = 2;
  arma::vec mean {89,16};
  arma::mat chol {{10, 1},{0, 2}};
  arma::mat covariance = chol.t()*chol;
  distribution::Gaussian<2> pdf(mean,covariance);


  std::vector<arma::vec> rvs(mc_n);

  random::setRandomSeed();

  arma::mat samples(dimension,mc_n);
  samples.each_col([&pdf](arma::vec &col){col = pdf.random();});
  auto mc_mean = arma::sum(samples,1)/mc_n;
  auto mc_covariance = arma::cov(samples.t());
  BOOST_CHECK(arma::approx_equal(mean, mc_mean, "absdiff", 0.1));
  BOOST_CHECK(arma::approx_equal(covariance, mc_covariance, "absdiff", 0.1));
}
BOOST_AUTO_TEST_SUITE_END();
