#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/filter/resampler/base.hpp"
#include "ssmpack/filter/resampler/systematic.hpp"

#include <armadillo>

using namespace ssmpack;

BOOST_AUTO_TEST_SUITE(filter_resampler_base);

BOOST_AUTO_TEST_CASE(operator_parenthesis) {
  struct AlwaysTrue {
    bool operator()(arma::vec t) { return true; }
  };

  auto resampler = filter::resampler::makeSystematic(AlwaysTrue());
  random::setRandomSeed();

  int N = 500;
  
  arma::umat pars(1,N);
  for (int i=0; i<N; ++i)
     pars(0,i) = i+1;

  arma::vec w = arma::ones<arma::vec>(N) * 1.0 / N;

  arma::vec w_r = w;
  decltype(pars) pars_r = pars;
  
  // uniform weights, nothing should change
  resampler(pars_r, w_r);
  BOOST_CHECK(arma::all(w_r == w));
  BOOST_CHECK(arma::all(arma::all(pars_r == pars)));

  // make first i weights non-zero, any particle with zero weight
  // should not appear in the output
  for (int i = 1; i < N-1; ++i) {
    w_r.zeros();
    w_r.head(i).fill(1.0 / i);

    pars_r = pars;
    resampler(pars_r, w_r);
    BOOST_CHECK(arma::all(w_r == w));

    for (int j = i + 1; j <= N; ++j)
      BOOST_CHECK(!arma::any(arma::vectorise(pars_r) == j));
  }
}

BOOST_AUTO_TEST_CASE(operator_parenthesis_no_action) {
  struct AlwaysFalse {
    bool operator()(arma::vec t) { return false; }
  };

  auto resampler = filter::resampler::makeSystematic(AlwaysFalse());
  random::setRandomSeed();

  int N = 500;
  
  arma::umat pars(1,N);
  for (int i=0; i<N; ++i)
     pars(0,i) = i+1;

  arma::vec w = arma::zeros<arma::vec>(N);
  w(0) = 1;

  arma::vec w_r = w;
  decltype(pars) pars_r = pars;
  
  // criterion is false, shouldn't do anything
  resampler(pars_r, w_r);
  BOOST_CHECK(arma::all(w_r == w));
  BOOST_CHECK(arma::all(arma::all(pars_r == pars)));

}

BOOST_AUTO_TEST_SUITE_END();
