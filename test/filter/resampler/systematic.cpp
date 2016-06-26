#include <boost/test/unit_test.hpp>
#include <iostream>

#include <armadillo>

#define protected public  // for testing protected member
#include "ssmpack/filter/resampler/systematic.hpp"

using namespace ssmpack;

BOOST_AUTO_TEST_SUITE(filter_resampler_systematic);

BOOST_AUTO_TEST_CASE(ordered_number_generator)
{
  struct AlwaysTrue {
    bool operator()(arma::vec t) { return true; }
  };

  auto resampler = filter::resampler::makeSystematic(AlwaysTrue());
  random::setRandomSeed();
  int N = 500;
  for (int i = 1; i < 100; ++i) {
    auto u = resampler.generateOrderedNumbers(N);
    // first element should be < 1/N
    BOOST_CHECK(u[0] < 1.0 / N);
    // all u elements should be in [0 1)
    BOOST_CHECK(arma::all(u >= 0.0));
    BOOST_CHECK(arma::all(u < 1.0));
    // the difference between elements should be 1/N
    BOOST_CHECK(arma::approx_equal(
        arma::diff(u), arma::ones<arma::vec>(N-1) * 1.0 / N, "absdiff", 0.001));
  }
}

BOOST_AUTO_TEST_SUITE_END();
