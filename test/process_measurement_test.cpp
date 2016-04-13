#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/model/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional_distribution.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/measurement.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(process_measurement_test);

BOOST_AUTO_TEST_CASE(test1) {
  model::LinearGaussian<2, 2> f{{{1, 1}, {0, 1}}, {{0.0001, 0}, {0, 0.0001}}};
  model::LinearGaussian<1, 2> h{{1, 0}, {0.0001}};
  auto dynamic_cpdf =
      distribution::makeParametericConditional(distribution::Gaussian<2>(), f);
  auto measurement_cpdf =
      distribution::makeParametericConditional(distribution::Gaussian<1>(), h);

  auto markov_p = process::makeMarkov(dynamic_cpdf);

  auto measurement_p = process::makeMeasurement(markov_p, measurement_cpdf);

  std::cout << "a \n " << measurement_p.initialize(distribution::Gaussian<2>()) <<
  std::endl;

auto twin =  measurement_p.random();

}

BOOST_AUTO_TEST_SUITE_END();
