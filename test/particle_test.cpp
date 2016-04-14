#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/distribution/particle.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(particle_test);

BOOST_AUTO_TEST_CASE(one_step_test) {

  distribution::Particle<int> p = {2, 18};
  BOOST_CHECK_EQUAL(p.weight, 18);
  BOOST_CHECK_EQUAL(p.point, 2);

  int N = 100;
  std::vector<distribution::Particle<int>> pars(N);
  std::fill_n(pars.begin(), N, p);
  auto sum = distribution::normalize_particles(pars);

  for (auto &par : pars)
    BOOST_CHECK_EQUAL(par.weight, 1.0 / N);

  double cnt = 0;
  std::generate_n(pars.begin(), N, [&cnt]() -> distribution::Particle<int> {
    return {1, ++cnt};
  });
  sum = distribution::normalize_particles(pars);
  BOOST_CHECK_EQUAL(sum, (N + 1.0) * N / 2.0);
}

BOOST_AUTO_TEST_SUITE_END();
