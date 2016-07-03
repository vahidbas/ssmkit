#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmkit/distribution/categorical.hpp"

using namespace ssmkit;

BOOST_AUTO_TEST_SUITE(distribution_categorical);

// Number of Monte-Carlo runs 
constexpr int mc_n = 500000;

BOOST_AUTO_TEST_CASE(mc_test_random_default_pdf)
{
  // testing default constructor

  // categorical distribution with only one category = 0
  distribution::Categorical pdf;
  arma::Col<unsigned int> samples(mc_n);
  random::setRandomSeed();
  samples.for_each([&pdf](auto &val){val = pdf.random();});
  // check if any sample is non zero
  BOOST_CHECK(!arma::any(samples));
}

BOOST_AUTO_TEST_CASE(mc_test_random_arbitary_pdf)
{
  // testing distribution with given probabilities
  arma::vec probabilities{0.1, 0.3, 0.05, 0.0, 0.55};
  distribution::Categorical pdf(probabilities);

  // making histogram of returned samples
  arma::vec hist{0, 0, 0, 0, 0};
  random::setRandomSeed();
  for(int ii=0; ii<mc_n; ++ii)
    ++hist(pdf.random());
  // normalize histogram
  auto hist_n = hist / mc_n;
  // check if normalized histogram is equal to the given probabilities
  BOOST_CHECK(arma::approx_equal(hist_n, probabilities, "absdiff", 0.005));
}

BOOST_AUTO_TEST_CASE(likelihood_test)
{
}

BOOST_AUTO_TEST_SUITE_END();
