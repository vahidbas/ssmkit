#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/distribution/gaussian.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace ssmpack;

BOOST_AUTO_TEST_SUITE(distribution_gausian);

constexpr int mc_n = 500000;

BOOST_AUTO_TEST_CASE(mc_test_random_default_pdf)
{
  constexpr int dimension = 8;
  distribution::Gaussian<dimension> pdf;

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

  random::setRandomSeed();

  arma::mat samples(dimension,mc_n);
  samples.each_col([&pdf](arma::vec &col){col = pdf.random();});

  auto mc_mean = arma::sum(samples,1)/mc_n;
  auto mc_covariance = arma::cov(samples.t());
  BOOST_CHECK(arma::approx_equal(mean, mc_mean, "absdiff", 0.1));
  BOOST_CHECK(arma::approx_equal(covariance, mc_covariance, "absdiff", 0.1));
}

BOOST_AUTO_TEST_CASE(likelihood_test)
{
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

  distribution::Gaussian<3> pdf;
  arma::mat ln_test(size(ln));
  auto it_ln_test = ln_test.begin();
  rvs.each_col([&pdf, &it_ln_test](arma::vec &col) {
    *it_ln_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(ln, ln_test, "absdiff", 0.001));
  
  pdf.parameterize(arma::zeros<arma::vec>(3), arma::eye<arma::mat>(3,3)*5.0);
  arma::mat ln5_test(size(ln5));
  auto it_ln5_test = ln5_test.begin();
  rvs.each_col([&pdf, &it_ln5_test](arma::vec &col) {
    *it_ln5_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(ln5, ln5_test, "absdiff", 0.001));

  pdf.parameterize({1,2,3}, arma::eye<arma::mat>(3,3)*5.0);
  arma::mat lnm_test(size(lnm));
  auto it_lnm_test = lnm_test.begin();
  rvs.each_col([&pdf, &it_lnm_test](arma::vec &col) {
    *it_lnm_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(lnm, lnm_test, "absdiff", 0.001));


  arma::mat chol {{1, 1, 1},{0, 1, 1},{0, 0, 1}};
  arma::mat covariance = chol.t()*chol;
  pdf.parameterize(arma::zeros<arma::vec>(3), covariance);
  arma::mat lnc_test(size(lnc));
  auto it_lnc_test = lnc_test.begin();
  rvs.each_col([&pdf, &it_lnc_test](arma::vec &col) {
    *it_lnc_test++ = pdf.likelihood(col);
  });
  BOOST_CHECK(arma::approx_equal(lnc, lnc_test, "absdiff", 0.001));
}

BOOST_AUTO_TEST_SUITE_END();
