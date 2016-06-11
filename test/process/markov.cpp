#include <boost/test/unit_test.hpp>
#include <iostream>

#include "ssmpack/map/linear_gaussian.hpp"
#include "ssmpack/distribution/gaussian.hpp"
#include "ssmpack/distribution/conditional.hpp"
#include "ssmpack/process/markov.hpp"
#include "ssmpack/process/memoryless.hpp"

#include "ssmpack/process/hierarchical.hpp"

using namespace PROJECT_NAME;

BOOST_AUTO_TEST_SUITE(markov_test);

BOOST_AUTO_TEST_CASE(test1) {
  arma::arma_rng::set_seed_random();

  map::LinearGaussian f{{{1, 1}, {0, 1}}, {{0.0001, 0}, {0, 0.01}}};
  distribution::Gaussian nu(2);

  map::LinearGaussian h{{1, 0}, {0.0001}};
  distribution::Gaussian omega(1);

  distribution::Conditional<decltype(nu), decltype(f)> dyn(nu, f);
  distribution::Conditional<decltype(omega), decltype(h)> obs(omega, h);

  process::Markov<decltype(nu), decltype(f), decltype(nu)> markov(dyn, nu);
  process::Memoryless<decltype(omega), decltype(h)> memless(obs);

  auto r = markov.initialize();
  auto r2 = memless.random(r);
  // double l = markov.likelihood(r);

  process::Hierarchical<decltype(markov), decltype(memless)> tt(markov, memless);

  //std::cout << process::ProcessTraits<decltype(markov)>::TArity::value
  //          << std::endl;

  tt.initialize();
  auto r3 = tt.random(1, 'c');
  auto ll = tt.likelihood(r3);

  //std::cout << std::get<0>(r3) << std::get<1>(r3) << std::endl;

    std::vector<decltype(markov.random())> v;
  
    markov.random_n(v, 10);

   // std::for_each(v.begin(), v.end(), [](decltype(markov.random()) &p) {
   //   std::cout << p << std::endl;
   // });
}

BOOST_AUTO_TEST_SUITE_END();
