#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmkit/distribution/gaussian.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace ssmkit;

BOOST_AUTO_TEST_SUITE(distribution_gausian);

// Number of Monte-Carlo runs 
constexpr int mc_n = 500000;

BOOST_AUTO_TEST_CASE(mc_test_random_default_pdf) {
  // testing distribution with default constructor

  constexpr int dimension = 8;
  // 8-dimensional Gaussian distribution with zero mean identity covariance
  distribution::Gaussian pdf(dimension);
  
  // set random seed
  random::setRandomSeed();
  
  // sampling large number of random variables from distribution
  arma::mat samples(dimension, mc_n);
  samples.each_col([&pdf](arma::vec &col) { col = pdf.random(); });
  // Calculating sample mean
  auto mc_mean = arma::sum(samples, 1) / mc_n;
  // Calculating sample covariance
  auto mc_covariance = arma::cov(samples.t());

  // check if sample mean is close to zero
  BOOST_CHECK(arma::approx_equal(mc_mean, arma::zeros<arma::vec>(dimension),
                                 "absdiff", 0.01));
  // check if sample covariance is close to identity
  BOOST_CHECK(arma::approx_equal(mc_covariance,
                                 arma::eye<arma::mat>(dimension, dimension),
                                 "absdiff", 0.01));
}

BOOST_AUTO_TEST_CASE(mc_test_random_arbitary_pdf) {
  // testing distribution with given mean and covariance

  // construction distribution
  constexpr int dimension = 2;
  arma::vec mean{89, 16};
  arma::mat chol{{10, 1}, {0, 2}};
  arma::mat covariance = chol.t() * chol;
  distribution::Gaussian pdf(mean, covariance);
  
  // set random seed
  random::setRandomSeed();

  // sampling large number of random variables from distribution
  arma::mat samples(dimension, mc_n);
  samples.each_col([&pdf](arma::vec &col) { col = pdf.random(); });
  // Calculating sample mean
  auto mc_mean = arma::sum(samples, 1) / mc_n;
  // Calculating sample covariance
  auto mc_covariance = arma::cov(samples.t());

  // check if sample mean is close to the given mean
  BOOST_CHECK(arma::approx_equal(mean, mc_mean, "absdiff", 0.1));
  // check if sample covariance is close to given covariance
  BOOST_CHECK(arma::approx_equal(covariance, mc_covariance, "absdiff", 0.5));
}

BOOST_AUTO_TEST_CASE(likelihood_test) {
  // check the likelihood function using precomputed data
  arma::mat rvs;
  arma::mat ln;
  arma::mat ln5;
  arma::mat lnm;
  arma::mat lnc;

  BOOST_REQUIRE(rvs.load(STR(SOURCE_DIR) "/test/data/gaussian/rvs.csv"));
  BOOST_REQUIRE(ln.load(STR(SOURCE_DIR) "/test/data/gaussian/ln.csv"));
  BOOST_REQUIRE(ln5.load(STR(SOURCE_DIR) "/test/data/gaussian/ln5.csv"));
  BOOST_REQUIRE(lnm.load(STR(SOURCE_DIR) "/test/data/gaussian/lnm.csv"));
  BOOST_REQUIRE(lnc.load(STR(SOURCE_DIR) "/test/data/gaussian/lnc.csv"));

  distribution::Gaussian pdf(3);
  arma::mat ln_test(size(ln));
  auto it_ln_test = ln_test.begin();
  rvs.each_col([&pdf, &it_ln_test](arma::vec &col) {
    *it_ln_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(ln, ln_test, "absdiff", 0.001));

  pdf.parameterize(arma::zeros<arma::vec>(3), arma::eye<arma::mat>(3, 3) * 5.0);
  arma::mat ln5_test(size(ln5));
  auto it_ln5_test = ln5_test.begin();
  rvs.each_col([&pdf, &it_ln5_test](arma::vec &col) {
    *it_ln5_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(ln5, ln5_test, "absdiff", 0.001));

  pdf.parameterize({1, 2, 3}, arma::eye<arma::mat>(3, 3) * 5.0);
  arma::mat lnm_test(size(lnm));
  auto it_lnm_test = lnm_test.begin();
  rvs.each_col([&pdf, &it_lnm_test](arma::vec &col) {
    *it_lnm_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(lnm, lnm_test, "absdiff", 0.001));

  arma::mat chol{{1, 1, 1}, {0, 1, 1}, {0, 0, 1}};
  arma::mat covariance = chol.t() * chol;
  pdf.parameterize(arma::zeros<arma::vec>(3), covariance);
  arma::mat lnc_test(size(lnc));
  auto it_lnc_test = lnc_test.begin();
  rvs.each_col([&pdf, &it_lnc_test](arma::vec &col) {
    *it_lnc_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(lnc, lnc_test, "absdiff", 0.001));
}

BOOST_AUTO_TEST_SUITE_END();
