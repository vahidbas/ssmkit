#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/distribution/categorical.hpp"

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

using namespace ssmpack;

BOOST_AUTO_TEST_SUITE(distribution_categorical);

constexpr int mc_n = 500000;

BOOST_AUTO_TEST_CASE(mc_test_random_default_pdf)
{
  distribution::Categorical<> pdf;
  arma::Col<int> samples(mc_n);
  random::setRandomSeed();
  samples.for_each([&pdf](int &val){val = pdf.random();});
  BOOST_CHECK(!arma::any(samples));
}

BOOST_AUTO_TEST_CASE(mc_test_random_arbitary_pdf)
{
  arma::vec probabilities{0.1, 0.3, 0.05, 0.0, 0.55};
  distribution::Categorical<> pdf(probabilities);
  arma::vec hist{0, 0, 0, 0, 0};

  random::setRandomSeed();
  for(int ii=0; ii<mc_n; ++ii)
    ++hist(pdf.random());

  auto hist_n = hist / mc_n;
  BOOST_CHECK(arma::approx_equal(hist_n, probabilities, "absdiff", 0.005));
}

BOOST_AUTO_TEST_CASE(likelihood_test)
{
}

BOOST_AUTO_TEST_SUITE_END();
