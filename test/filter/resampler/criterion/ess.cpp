#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/filter/resampler/criterion/ess.hpp"

using namespace ssmpack;

BOOST_AUTO_TEST_SUITE(filter_resampler_criterion_ess);

BOOST_AUTO_TEST_CASE(operator_parenthesis)
{
  filter::resampler::criterion::ESS crt(9.9);

  arma::vec w(10);
  w.fill(0.1);
  // w is 100% effective
  BOOST_CHECK(!crt(w));
  
  w.zeros();
  w(0) = 1;
  // only one effective
  BOOST_CHECK(crt(w));

  crt.th = 2;
  w(0) = 0.25; w(1) = 0.25; w(2) = 0.25; w(3) = 0.25; 
  // four effectives
  BOOST_CHECK(!crt(w));

  crt.th = 4.1;
  // four effectives
  BOOST_CHECK(crt(w));
}

BOOST_AUTO_TEST_SUITE_END();
